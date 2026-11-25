import torch

from hla_tok import HLATok
from hla_tok.streaming.continuous import (
    ContinuousReceiver,
    RankOrder,
    encode_continuous,
    stream_frames,
    truncate,
)
from scripts.hla_tok.continuous.dynamic import plans
from scripts.hla_tok.v2.protocol import setup


def test_current_frame_can_be_emitted_before_future_input_is_read():
    setup()
    torch.manual_seed(4)
    model = HLATok("R2").eval()
    frame = torch.rand(1, 3, 1080, 1920)
    order = RankOrder.native()

    def source():
        yield frame, 1
        raise AssertionError("future input requested before current frame was emitted")

    prefix, image = next(stream_frames(model, source(), order))
    expected = truncate(encode_continuous(model, frame), 1)
    assert torch.equal(prefix, expected)
    assert torch.equal(image, ContinuousReceiver.from_model(model).decode(expected, order))


def test_budget_switches_do_not_carry_reconstruction_into_next_frame():
    setup()
    torch.manual_seed(19)
    model = HLATok("R2").eval()
    past = torch.rand(1, 3, 1080, 1920)
    current = 1 - past
    order = RankOrder.reversed()
    left = list(stream_frames(model, [(past, 1), (current, 510)], order))
    right = list(stream_frames(model, [(past, 6120), (current, 510)], order))
    assert not torch.equal(left[0][1], right[0][1])
    assert all(torch.equal(a, b) for a, b in zip(left[1], right[1], strict=True))


def test_six_schedules_are_fixed_at_clip_boundaries():
    config = {"seed": 23, "constant_budget": 4}
    schedules = plans([1, 4, 8], 12, config, 0)
    assert tuple(schedules) == ("constant", "descending", "ascending", "alternate", "random", "recover")
    assert schedules["descending"] == [8] * 4 + [4] * 4 + [1] * 4
    assert schedules["ascending"] == schedules["descending"][::-1]
    assert schedules["recover"] == [1] * 8 + [8] * 4
    assert schedules == plans([1, 4, 8], 12, config, 0)
    assert schedules["random"] != plans([1, 4, 8], 12, config, 1)["random"]
