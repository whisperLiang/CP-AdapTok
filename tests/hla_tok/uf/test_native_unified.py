import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_native_unified_prefix_uses_only_transmitted_quantized_latent(native_assets):
    import torch

    from hla_tok.uf.anchor import BUDGETS
    from hla_tok.uf.latent import UnifiedCodec, UnifiedRefinement, observe
    from hla_tok.uf.official import NativeDecoder
    from scripts.hla_tok.temporal.inputs import source_frame
    from scripts.hla_tok.uf.inputs import inputs
    from scripts.hla_tok.uf.unified_eval import UnifiedEncoder

    codec = UnifiedCodec(UnifiedRefinement().cuda().eval())
    encoder = UnifiedEncoder(native_assets, codec, (0, 8))
    baseline, enhanced = NativeDecoder(native_assets), NativeDecoder(native_assets)
    reader, plan = inputs()
    clip = plan["calibration"][0]
    for index in range(3):
        encoded = encoder.encode(source_frame(reader, clip, index), index, index == 0)
        family = encoded.family
        prefix, records = family.prefix(BUDGETS[5])
        stable = encoder.receiver(baseline)
        stable.feed(family.raw[:family.base_end])
        assert stable.decode_available() == 1
        varied = encoder.receiver(enhanced)
        for start in range(0, len(prefix), 131):
            varied.feed(prefix[start:start + 131])
            varied.decode_available()
        varied.finish()
        assert varied.canvas.records == records > 0
        assert torch.equal(stable.render(), varied.base)
        assert encoder.matches(encoded, enhanced, varied)
        assert torch.isfinite(varied.render()).all()
        if index:
            first = observe(baseline, False, "uf_state_snapshot")
            second = observe(enhanced, False, "uf_state_snapshot")
            assert all(torch.equal(left, right) for left, right in zip(first, second, strict=True))
    assert reader.counts["test"] == 0
