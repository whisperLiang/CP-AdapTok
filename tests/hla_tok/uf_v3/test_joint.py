import numpy as np
import torch

from hla_tok.uf_v3.joint import packed_prefix, refinement


def test_real_record_prefix_only_drops_a_suffix_and_keeps_gradient():
    from hla_tok.uf.anchor import Segmentation
    from hla_tok.uf_v2.records import BatchRecords

    symbols = np.ones((2, 8, 8, 8), dtype=np.int8)
    bank = BatchRecords((1., .5), Segmentation(8, 8))
    low, high = packed_prefix(bank, True, symbols, (50, 2048))
    assert low.records == 0 and high.records == 2
    assert high.bytes_used > low.bytes_used
    assert np.all(high.symbols == 3) and np.all(high.levels == 2)

    initial = torch.full((1, 8, 8, 8), .148, requires_grad=True)
    step = torch.full((1, 8, 1, 1), .125, requires_grad=True)
    coarse, fine = refinement(initial, step)
    assert torch.equal(coarse / step, coarse.new_ones(coarse.shape))
    assert torch.count_nonzero(fine) == 0
    loss = (coarse + fine).square().mean()
    loss.backward()
    assert initial.grad is not None and torch.count_nonzero(initial.grad) == initial.numel()
    assert step.grad is not None and torch.count_nonzero(step.grad) == step.numel()


def test_joint_probe_rejects_frozen_encoder_and_a_missing_rate_path():
    import pytest

    from scripts.hla_tok.uf_v3.train import verify_joint_gradients

    parameters = {name: torch.ones((), requires_grad=True) for name in ("intra_encoder", "inter_encoder")}
    healthy = {term: dict.fromkeys(parameters, 0.1) for term in ("distortion", "rate")}
    verify_joint_gradients(parameters, healthy)
    rates = healthy | {"rate": healthy["rate"] | {"inter_encoder": 0.0}}
    with pytest.raises(RuntimeError, match="rate lacks a valid inter_encoder gradient"):
        verify_joint_gradients(parameters, rates)
    parameters["inter_encoder"].requires_grad_(False)
    with pytest.raises(RuntimeError, match="was frozen"):
        verify_joint_gradients(parameters, healthy)


def test_training_accepts_raw_rgb_and_rejects_cached_latent_or_feature():
    import pytest

    from scripts.hla_tok.uf_v3.train import require_raw_rgb

    require_raw_rgb(torch.zeros(4, 3, 128, 192))
    for forbidden in (torch.zeros(4, 128, 8, 12), torch.zeros(4, 256, 16, 24), torch.zeros(4, 3, 128, 192).byte()):
        with pytest.raises(ValueError, match="consecutive floating-point RGB"):
            require_raw_rgb(forbidden)


def test_rate_remains_differentiable_near_real_integer_symbols():
    from hla_tok.uf.entropy import cdf_table
    from hla_tok.uf_v3.joint import conditional_cost, static_cost

    signed = torch.ones(1, 8, 2, 2, requires_grad=True)
    static = static_cost(signed, (cdf_table(1.),), 0)
    conditional = conditional_cost(signed, torch.tensor(1.))
    for cost in (static, conditional):
        gradient, = torch.autograd.grad(cost.sum(), signed, retain_graph=True)
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) == signed.numel()


def test_raw_rgb_unroll_routes_separate_encoder_gradients_and_keeps_base_state():
    from pathlib import Path

    import pytest

    from hla_tok.uf.latent import UnifiedRefinement
    from hla_tok.uf_v2.decoder import SharedRefinement
    from hla_tok.uf_v2.records import BatchRecords
    from hla_tok.uf_v3.joint import SEGMENTS, DifferentiableUF, progressive_frame
    from third_party.DCVC.src.models.image_model import DMCI
    from third_party.DCVC.src.models.video_model_ld import DMC
    from third_party.DCVC.src.utils.common import get_state_dict

    weights = Path("checkpoints")
    paths = tuple(weights / name for name in ("cvpr2026_image.pth.tar", "cvpr2026_video_ld.pth.tar"))
    if not all(path.exists() for path in paths):
        pytest.skip("verified native backbone weights are needed for the I/P graph regression")
    model = DifferentiableUF(DMCI(), DMC(), UnifiedRefinement(), SharedRefinement()).requires_grad_(False)
    for net, path in zip((model.intra, model.inter), paths, strict=True):
        net.load_state_dict(get_state_dict(path))
    parameters = (model.intra.enc.enc_2[-1].weight, model.inter.encoder.down.weight, model.inter.decoder.conv2.weight)
    for parameter in parameters:
        parameter.requires_grad_()
    colors = torch.rand(2, 3, 64, 64, generator=torch.Generator().manual_seed(333))
    bank = BatchRecords((1., .5), SEGMENTS)
    count = 0

    def count_inter(*_args):
        nonlocal count
        count += 1

    hook = model.inter.encoder.register_forward_hook(count_inter)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        first = progressive_frame(model, bank, colors[0:1], None)
        second = progressive_frame(model, bank, colors[1:2], first.state)
    hook.remove()
    assert count == 1
    assert first.state.memory is None and second.state.memory is not None
    assert first.rgb[0].shape == second.rgb[0].shape == (1, 3, 64, 64)
    assert second.physical_enhancement_bytes == sorted(second.physical_enhancement_bytes)
    dist = (first.rgb[-1] - colors[:1]).square().mean() + (second.rgb[-1] - colors[1:]).square().mean()
    rate = first.rates[-1] + second.rates[-1]
    d = torch.autograd.grad(dist, parameters, retain_graph=True)
    r = torch.autograd.grad(rate, parameters[:2])
    assert all(torch.isfinite(grad).all() and float(grad.norm()) > 0 for grad in (*d, *r))


def test_native_prefix_rejects_unsorted_or_negative_caps():
    import pytest

    from hla_tok.uf.anchor import Segmentation
    from hla_tok.uf_v2.records import BatchRecords

    bank = BatchRecords((1., .5), Segmentation(8, 8))
    symbols = np.ones((2, 8, 8, 8), dtype=np.int8)
    for invalid in ((), (3, 0), (-1, 2)):
        with pytest.raises(ValueError, match="enhancement allowances"):
            packed_prefix(bank, True, symbols, invalid)
    assert packed_prefix(bank, True, symbols, (0,))[0].records == 0
