import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.integration]


def test_independent_decoders_agree_on_native_pictures(native_assets):
    import torch

    from hla_tok.uf.official import NativeDecoder, NativeEncoder, NativeNAL
    from scripts.hla_tok.temporal.inputs import source_frame
    from scripts.hla_tok.uf.inputs import inputs

    encoder = NativeEncoder(native_assets)
    preview, destination = NativeDecoder(native_assets), NativeDecoder(native_assets)
    framer, parser = NativeNAL(), NativeNAL()
    reader, plans = inputs()
    clip = plans["calibration"][0]
    for index in range(6):
        picture = source_frame(reader, clip, index)
        original = encoder.encode(picture, index, 0 if index == 0 else 8, index == 0)
        received = parser.decode(framer.encode(original), index)
        assert received == original
        local = preview.decode(original)
        remote = destination.decode(received)
        assert torch.equal(local, remote), f"RGB mismatch at frame {index}"
        assert torch.equal(preview.decoded_latent(), destination.decoded_latent()), index
    assert reader.counts["test"] == 0


def _reference_tensors(decoder):
    import torch

    net = decoder.last_net
    with torch.cuda.stream(decoder.stream):
        snapshot = getattr(net.proxy, "uf_state_snapshot", None)
        values = [net.proxy.get_decoded_latent()] if snapshot is None else snapshot()
    decoder.stream.synchronize()
    return values


def _same_reference(left, right):
    import torch

    return all(torch.equal(expected, actual) for expected, actual in zip(left, right, strict=True))


def _feed_random_fragments(receiver, raw, rng):
    from itertools import accumulate

    sizes = (rng.randrange(1, 1025) for _ in range(len(raw)))
    start = 0
    for end in accumulate(sizes):
        receiver.feed(raw[start:end])
        receiver.decode_available()
        if end >= len(raw):
            return
        start = end


def test_native_anchor_growth_budget_switch_loss_and_future_perturbation(native_assets):
    from random import Random

    import torch

    from hla_tok.uf.anchor import BUDGETS, AnchorModel, ProgressiveReceiver, RefinementCodec, decode_prefix
    from hla_tok.uf.official import NativeDecoder
    from scripts.hla_tok.temporal.inputs import source_frame
    from scripts.hla_tok.uf.anchor_eval import AnchorEncoder
    from scripts.hla_tok.uf.inputs import inputs

    codec = RefinementCodec(AnchorModel().cuda().eval(), "bf16")
    encoder = AnchorEncoder(native_assets, codec, (0, 8))
    stable, changing, dropped = (NativeDecoder(native_assets) for _ in range(3))
    reader, plans = inputs()
    clip = plans["calibration"][0]
    family_history = []
    rng = Random(91)
    for index, bandwidth in enumerate((5, 40, 10, 20, 5, 10)):
        picture = source_frame(reader, clip, index)
        encoded = encoder.encode(picture, index, index % 4 == 0)
        family = encoded.family
        family_history.append(family.raw)
        base, _ = family.prefix(family.base_end)
        prefix, records = family.prefix(BUDGETS[bandwidth])
        assert family.raw.startswith(prefix) and prefix.startswith(base) and records > 0
        stable_frame = decode_prefix(codec, stable.decode, base)
        assert torch.equal(stable_frame.render(), encoded.reference)
        reference = _reference_tensors(stable)
        receiver = ProgressiveReceiver(codec, changing.decode)
        _feed_random_fragments(receiver, base, rng)
        assert torch.equal(receiver.render(), encoded.reference)
        assert _same_reference(reference, _reference_tensors(changing))
        _feed_random_fragments(receiver, prefix[len(base):], rng)
        receiver.finish()
        assert receiver.canvas.records == records
        assert torch.isfinite(receiver.render()).all()
        assert _same_reference(reference, _reference_tensors(changing))
        if index == 3:
            with pytest.raises(ValueError, match="DESYNC"):
                decode_prefix(codec, dropped.decode, base)
        elif index != 2:
            complete = decode_prefix(codec, dropped.decode, prefix)
            assert torch.equal(complete.render(), receiver.render())
            assert _same_reference(reference, _reference_tensors(dropped))

    encoder.reset()
    for index in range(6):
        picture = source_frame(reader, clip, index)
        modified = 1 - picture if index >= 2 else picture
        encoded = encoder.encode(modified, index, index % 4 == 0)
        assert (encoded.family.raw == family_history[index]) == (index < 2)
    assert reader.counts["test"] == 0
