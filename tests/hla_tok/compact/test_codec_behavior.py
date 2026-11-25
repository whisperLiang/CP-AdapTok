import pytest
import torch

from hla_tok.compression.profiles import Profile
from hla_tok.compression.scalar import StaticScalar
from hla_tok.compression.transform import CompactTransform
from hla_tok.streaming import compact
from hla_tok.streaming.compact import CompactCodec, encode_frames
from hla_tok.streaming.compact_wire import deserialize
from hla_tok.streaming.continuous import MAX_K, RankOrder


@pytest.fixture
def codec():
    order = RankOrder.reversed()
    transform = CompactTransform.from_pca(torch.zeros(1, 32), torch.eye(32)[None, :, :4])
    scalar = StaticScalar(torch.ones((1, 4)), 4)
    return CompactCodec(Profile.Q4_4, transform, scalar, order, "aa" * 32, "bb" * 32, "cpu")


def test_prefix_excludes_untransmitted_token_information(codec):
    values = torch.zeros(1, MAX_K, 32)
    cut = 170
    ordered = codec.order.apply(values)
    ordered[:, :cut] = 1
    ordered[:, cut:] = -1
    source = codec.order.restore_full(ordered)
    changed = ordered.clone()
    changed[:, cut:] = 2
    altered = codec.order.restore_full(changed)
    a = codec.frame(codec.encode_tokens(source, cut), 1)
    b = codec.frame(codec.encode_tokens(altered, cut), 1)
    assert a.payload == b.payload == b"\x11\x11" * cut
    full = codec.frame(codec.encode_tokens(source, MAX_K), 1)
    assert full.payload[:len(a.payload)] == a.payload
    assert full.payload[len(a.payload):] == b"\xff\xff" * (MAX_K - cut)


def test_frame_interface_has_no_history_and_changes_budget_immediately(codec, monkeypatch):
    def tokens(model, frame, budget=MAX_K):
        return frame.reshape(1, 1, 1).expand(1, budget, 32)

    monkeypatch.setattr(compact, "encode_continuous", tokens)
    sequence = [torch.tensor(-2.), torch.tensor(2.), torch.tensor(-6.)]
    counts = [3, 51, 1]

    def stream(frames, budgets):
        return [packet.serialize_range(0, packet.count)
                for packet in encode_frames(None, codec, zip(frames, budgets, strict=True))]

    a = stream(sequence, counts)
    b = stream([*sequence[:2], torch.tensor(5.)], counts)
    assert a[:2] == b[:2]
    assert a[-1] != b[-1]
    for i, k in enumerate(counts):
        q = codec.unpack(deserialize(a[i]).payload)
        assert q.shape == (1, k, 4)
        assert q.eq(sequence[i].item()).all()


def test_stream_yields_before_reading_future_frames(codec, monkeypatch):
    monkeypatch.setattr(compact, "encode_continuous", lambda *args: torch.zeros(1, MAX_K, 32))
    def inputs():
        yield torch.zeros(1), 1
        raise AssertionError("future frame requested before first output")
    assert next(encode_frames(None, codec, inputs())).count == 1
