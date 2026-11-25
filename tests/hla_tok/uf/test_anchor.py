from random import Random

import pytest
import torch

from hla_tok.uf.anchor import AnchorModel, ProgressiveReceiver, RefinementCodec, Segmentation
from hla_tok.uf.base import BaseFrame, FrameInfo
from hla_tok.uf.wire import PrefixFamily


def test_rgb_analysis_and_exact_no_enhancement_path():
    model = AnchorModel()
    rgb = torch.linspace(0, 1, 3 * 64 * 48).view(1, 3, 64, 48)
    base = torch.full_like(rgb, 0.5)
    latent = model.analyze(rgb, base)
    assert latent.shape == (1, model.channels, 8, 6)
    coarse, fine = model.quantizer(latent)
    assert model.display(base, coarse + fine).sub(rgb).square().mean() < base.sub(rgb).square().mean()


def test_static_cdf_scan_orders_coarse_records_before_fine():
    pytest.importorskip("hla_uf_rans")
    codec = RefinementCodec(AnchorModel(), "fp32", Segmentation(2, 12, conditional=False, ranked=False))
    coarse = torch.ones((1, 12, 2, 4), dtype=torch.int8)
    fine = -coarse
    records = list(codec.records_codec.records(coarse, fine))
    assert [(row.level, row.group, row.cdf) for row in records] == [(0, 0, 1), (0, 1, 1), (1, 0, 5), (1, 1, 5)]


@pytest.mark.integration
def test_receiver_applies_real_symbols_once_across_random_append_and_preserves_base():
    pytest.importorskip("hla_uf_rans")
    torch.manual_seed(2)
    model = AnchorModel()
    codec = RefinementCodec(model, "fp32")
    frame = BaseFrame(FrameInfo(0, 144, 144, 0, True, 1, False), b"simulated base for unit test")
    base = torch.full((1, 3, 144, 144), 0.4)
    original = torch.linspace(0, 1, base.numel()).view(base.shape)
    decoded = []

    def decode(item):
        decoded.append(item)
        return base

    latent = model.analyze(original, base)
    family = PrefixFamily.build(frame, codec.tag, codec.records(latent))
    receiver = ProgressiveReceiver(codec, decode)
    receiver.feed(family.raw[:family.base_end])
    receiver.decode_available()
    assert receiver.render() is base
    assert receiver.symbol_count == 0
    rng = Random(42)
    start = family.base_end
    while start < len(family.raw):
        end = start + rng.randint(1, 131)
        receiver.feed(family.raw[start:end])
        receiver.decode_available()
        assert receiver.decode_available() == 0
        start = end
    receiver.finish()
    full = receiver.render()
    from hla_tok.uf.anchor import decode_prefix

    complete = decode_prefix(codec, lambda _: base, family.raw)
    assert torch.equal(full, complete.render())
    assert receiver.symbol_count == complete.symbol_count > 0
    assert torch.equal(base, torch.full_like(base, 0.4))
    assert decoded == [frame]
    assert (full - original).square().mean() < (base - original).square().mean()
