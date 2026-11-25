import numpy as np
import pytest
import torch

from hla_tok.compression.transform import CompactTransform
from hla_tok.models.codec import HLATok
from hla_tok.streaming.continuous import ContinuousReceiver, RankOrder
from hla_tok.temporal.layout import LAYOUT
from hla_tok.temporal.state import DisplayReconstructionState
from hla_tok.temporal.transform import NativeTransform, TemporalSynthesis

RANKS = LAYOUT.mask.shape[0]
DIM = LAYOUT.components


def factors(rank_aware=True):
    rng = torch.Generator().manual_seed(416)
    center = torch.randn(RANKS, DIM, generator=rng) * 0.2
    matrix = torch.randn(RANKS, DIM, DIM, generator=rng)
    basis = torch.linalg.qr(matrix).Q if rank_aware else torch.eye(DIM).repeat(RANKS, 1, 1)
    return center, basis


def test_rank_aware_projection_matches_cp_and_inverts():
    center, basis = factors()
    transform = NativeTransform(center, basis)
    reference = CompactTransform.from_pca(center, basis)
    values = torch.randn(2, LAYOUT.token_count, DIM, generator=torch.Generator().manual_seed(7))
    ranks = torch.tensor(LAYOUT.ranks, dtype=torch.long)
    coefficients = transform.project(values)
    torch.testing.assert_close(coefficients, reference.encode(values, ranks), atol=5e-6, rtol=4e-5)
    torch.testing.assert_close(transform.lift(coefficients), values, atol=7e-6, rtol=5e-5)


@pytest.mark.gpu
def test_full_support_bypass_and_masked_values_match_native_decoder():
    torch.manual_seed(29)
    model = HLATok("R2").eval().requires_grad_(False).cuda().to(memory_format=torch.channels_last)
    center, basis = factors()
    center, basis = center.cuda(), basis.cuda()
    zeros = torch.ones_like(center)
    temporal = TemporalSynthesis.from_model(model, NativeTransform(center, basis), zeros)
    original = ContinuousReceiver.from_model(model)
    native = torch.randn(1, LAYOUT.token_count, DIM, device="cuda")
    transform = temporal.transform
    coefficients = transform.project(native)
    support = np.ones(LAYOUT.token_count, dtype=bool)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = original.decode(native, RankOrder.native())
        observed = temporal.render_coefficients(coefficients, support)
        assert expected.shape == observed.shape == (1, 3, 1080, 1920)
        torch.testing.assert_close(expected, observed, rtol=0, atol=0.005)
        base = np.zeros(LAYOUT.token_count, dtype=bool)
        base[:135] = True
        sample = np.arange(LAYOUT.token_count * DIM, dtype=np.int16).reshape(-1, DIM) % 5
        altered = sample.copy()
        altered[~base] = 1000
        a = temporal.render(DisplayReconstructionState(sample, base, 1))
        b = temporal.render(DisplayReconstructionState(altered, base, 1))
        assert torch.equal(a, b)
        assert not hasattr(temporal, "encoder") and not hasattr(temporal, "query_encoder")


def test_inference_recovery_plan_can_be_reused_for_training_gradients():
    from hla_tok.temporal.layout import NativeLayout

    layout = NativeLayout()
    support = np.zeros(layout.token_count, dtype=bool)
    support[:140] = True
    mask = support.tobytes()
    with torch.inference_mode():
        old = torch.ones(1, layout.token_count, DIM)
        assert len(list(layout.recovery_groups(old, mask))) == 2
    current = torch.full((1, layout.token_count, DIM), 2.0, requires_grad=True)
    value = sum(group.square().sum() for group, _ in layout.recovery_groups(current, mask))
    value.backward()
    assert current.grad is not None
    assert torch.all(current.grad[0, support] == 4)
    assert torch.all(current.grad[0, ~support] == 0)
