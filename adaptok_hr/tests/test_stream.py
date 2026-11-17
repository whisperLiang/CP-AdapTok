from dataclasses import replace

import pytest
import torch


def test_65_frame_closed_loop_gop_missing_stream_and_truncated_history(small_model):
    from adaptok_hr.codec.bitstream import parse, serialize, truncate_packet
    from adaptok_hr.codec.codec import Codec

    codec = Codec(small_model.eval())
    frame = torch.rand(1, 3, 128, 256)
    sending, receiving = None, None
    kinds = []
    previous, second, first_packet, old_state = None, None, None, None
    for index in range(65):
        # Source changes each frame. Neither endpoint gets an original past frame in its state.
        current = (frame + index / 64).fmod(1)
        packet, sending = codec.encode_frame(current, sending, 32, "long")
        rgb, receiving = codec.decode_frame(packet, receiving)
        assert torch.equal(sending.rgb, receiving.rgb)
        assert torch.equal(sending.feature, receiving.feature)
        assert receiving.frame_index == index
        kinds.append(receiving.frame_type)
        if index == 0:
            first_packet = packet
        if index == 1:
            previous, second = old_state, packet
        if index % 32 == 0:
            reset, _ = codec.decode_frame(packet, None)
            assert torch.equal(reset, rgb)
        old_state = receiving
    assert [i for i, kind in enumerate(kinds) if kind == "I"] == [0, 32, 64]
    with pytest.raises(ValueError, match="stream"):
        codec.decode_frame(second, replace(previous, stream_id=b"z" * 16))
    with pytest.raises(ValueError, match="reference"):
        codec.decode_frame(second, replace(previous, chain=b"z" * 32))
    with pytest.raises(ValueError, match="version"):
        codec.decode_frame(second, replace(previous, version=2))
    with pytest.raises(ValueError):
        codec.decode_frame(serialize(replace(parse(second), frame_index=3)), previous)
    _, short_history = codec.decode_frame(truncate_packet(first_packet, 4), None)
    with pytest.raises(ValueError, match="reference"):
        codec.decode_frame(second, short_history)


def test_tail_codes_cannot_be_smuggled_through_decoder(small_model):
    from adaptok_hr.codec.bitstream import truncate_packet
    from adaptok_hr.codec.codec import Codec

    codec = Codec(small_model.eval())
    frame = torch.rand(1, 3, 128, 256)
    packet, _ = codec.encode_frame(frame, None, 32)
    short = truncate_packet(packet, 4)
    # Disturb every sender-side cached computation. Receiver result depends only on bytes.
    expected, _ = codec.decode_frame(short, None)
    small_model.encode_features(1 - frame, None)
    actual, _ = codec.decode_frame(short, None)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert len(short) < len(packet)


def test_conv_only_and_fixed_training_ablations(small_config):
    from adaptok_hr.models.model import AdapTokHR
    from adaptok_hr.tests.test_training import settings
    from adaptok_hr.training.train import JointTrainer

    config = replace(small_config, attention_blocks=0, temporal=False)
    model = AdapTokHR(config)
    plan = settings()
    plan["budget_mode"] = "fixed"
    trainer = JointTrainer(model, plan)
    row = trainer.update(torch.rand(3, 3, 128, 256), 1)
    assert row["budgets"] == [32, 32, 32]
    assert all(r["auxiliary_budget"] is None for r in row["frame_metrics"])
    assert all(v > 0 for v in row["module_update_norms"].values())
