import os
from pathlib import Path

import pytest
import torch

from baselines.dcvc_uf.official import OfficialAssets
from baselines.hla_tok_uf_v2.decoder import SharedRefinement
from baselines.hla_tok_uf_v2.latent import BUDGETS, UnifiedRefinement
from baselines.hla_tok_uf_v2.runtime import Unified
from baselines.hla_tok_uf_v2.wire import WireError
from common.inputs import inputs, setup, source_frame

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def codec_and_frames():
    directory = os.environ.get("UF_OFFICIAL_WEIGHTS")
    if not torch.cuda.is_available() or not directory:
        pytest.skip("requires the pinned UF-LD weights and a CUDA device")
    setup()
    assets = OfficialAssets.verify(Path(directory))
    reader, plans = inputs()
    frames = [source_frame(reader, plans["validation"][0], i) for i in range(4)]
    quantizer = UnifiedRefinement().cuda().eval()
    head = SharedRefinement().cuda().half().to(memory_format=torch.channels_last).eval()
    with torch.no_grad():
        for branch in (head.intra, head.inter):
            branch.out.weight.normal_(std=0.01)
    return Unified(assets, quantizer, (0, 8), head), frames


def ingest(codec, decoder, family, cap):
    receiver = codec.receiver(decoder)
    prefix, records = family.prefix(cap)
    receiver.feed(prefix)
    assert receiver.decode_available() == records + 1
    receiver.finish()
    return receiver


@torch.no_grad()
def test_native_d1_caches_current_features_and_never_updates_reference_with_display(codec_and_frames):
    codec, images = codec_and_frames
    codec.reset()
    base_endpoint, enhanced_endpoint = codec.new_decoder(), codec.new_decoder()
    retained = None
    for i, rgb in enumerate(images[:3]):
        encoded = codec.encode(rgb, i, i == 0)
        family = encoded.family
        base = ingest(codec, base_endpoint, family, family.base_end)
        display = ingest(codec, enhanced_endpoint, family, BUDGETS[5])
        zero = base.render()
        assert zero.shape == (1, 3, 1080, 1920) and torch.equal(zero, display.base)
        assert codec.matches(encoded, enhanced_endpoint, display)
        if i:
            before = enhanced_endpoint.video.proxy.uf_state_snapshot()
        shown = display.render()
        assert torch.isfinite(shown).all()
        assert not torch.equal(shown, zero)
        assert torch.equal(zero, display.base)
        if i:
            after = enhanced_endpoint.video.proxy.uf_state_snapshot()
            expected = base_endpoint.video.proxy.uf_state_snapshot()
            for a, b, c in zip(before, after, expected, strict=True):
                assert torch.equal(a, b)
                assert torch.equal(a, c)
        if i == 1:
            retained = display, shown.clone()
            display.feed(family.raw[family.ends[display.canvas.records]:family.ends[display.canvas.records + 1]])
            assert display.decode_available() == 1
            retained = display, display.render()
    assert retained is not None
    old, image = retained
    assert torch.equal(old.render(), image)
    assert enhanced_endpoint.order.last.index == 2


@torch.no_grad()
def test_native_future_change_loss_recovery_and_family_isolation(codec_and_frames):
    codec, images = codec_and_frames

    def generate(future):
        codec.reset()
        return [codec.encode(rgb, i, i in (0, 3)).family
                for i, rgb in enumerate([*images[:2], future, images[3]])]

    left = generate(images[2])
    right = generate(1 - images[2])
    assert left[:2] == right[:2]
    assert left[2].raw != right[2].raw
    decoder = codec.new_decoder()
    broken = codec.receiver(decoder)
    broken.feed(left[0].raw[:left[0].base_end - 1])
    assert broken.decode_available() == 0
    with pytest.raises(WireError, match="incomplete"):
        broken.finish()
    with pytest.raises(WireError, match="DESYNC"):
        ingest(codec, decoder, left[1], left[1].base_end)
    with pytest.raises(WireError, match="DESYNC"):
        ingest(codec, decoder, left[2], left[2].base_end)
    recovered = ingest(codec, decoder, left[3], left[3].base_end)
    assert recovered.render().shape == (1, 3, 1080, 1920)
    other = Unified(codec.assets, codec.model, (0, 0), codec.refiner, base_cap=4096, family_id="lite")
    with pytest.raises(WireError, match="model"):
        ingest(other, other.new_decoder(), left[3], left[3].base_end)


@torch.no_grad()
def test_adjusted_native_cdf_and_base_match_after_proxy_reload(codec_and_frames):
    import numpy as np

    from baselines.hla_tok_uf_v2.native_weights import NativeAdjustments

    codec, images = codec_and_frames
    delta = NativeAdjustments().cuda()
    delta.encoder_logits.fill_(0.05)
    delta.hyper_logits[19] = 5.0
    delta.reconstruction_logits.fill_(0.03)
    adjusted = Unified(codec.assets, codec.model, (0, 8), codec.refiner, adjustments=delta)
    endpoint = adjusted.new_decoder()
    candidate_table = adjusted.native.video.bit_estimator_z.get_cdf_info()
    decoder_table = endpoint.video.bit_estimator_z.get_cdf_info()
    for expected, observed in zip(candidate_table, decoder_table, strict=True):
        np.testing.assert_array_equal(observed, expected)
    assert not np.array_equal(candidate_table[0], codec.native.video.bit_estimator_z.get_cdf_info()[0])
    for i, rgb in enumerate(images[:3]):
        encoded = adjusted.encode(rgb, i, i == 0, reset=i == 1)
        receiver = ingest(adjusted, endpoint, encoded.family, BUDGETS[5])
        assert adjusted.matches(encoded, endpoint, receiver)
        assert torch.isfinite(receiver.render()).all()
        if i == 1:
            with pytest.raises(WireError, match="DESYNC"):
                ingest(adjusted, adjusted.new_decoder(), encoded.family, BUDGETS[5])


@torch.no_grad()
def test_a_codec_with_no_encoded_tensors_receives_same_display(codec_and_frames):
    codec, images = codec_and_frames
    codec.reset()
    isolated = Unified(codec.assets, codec.model, codec.qp, codec.refiner)
    participating, receiver_only = codec.new_decoder(), isolated.new_decoder()
    for i, rgb in enumerate(images[:2]):
        encoded = codec.encode(rgb, i, i == 0)
        expected = ingest(codec, participating, encoded.family, BUDGETS[5]).render()
        observed = ingest(isolated, receiver_only, encoded.family, BUDGETS[5]).render()
        assert torch.equal(observed, expected)
