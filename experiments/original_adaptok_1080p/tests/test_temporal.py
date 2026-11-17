import pytest
import torch

from experiments.original_adaptok_1080p.video_reader import temporal_windows


@pytest.mark.parametrize("length", [1, 15, 16, 17, 33])
def test_consecutive_windows_repeat_last_and_exclude_padding(length):
    clip = torch.arange(length).reshape(length, 1, 1, 1).expand(-1, 3, 2, 2).float()
    windows = list(temporal_windows(clip, 16, start=7))
    real = torch.cat([w.frames[:w.valid_frames] for w in windows])
    assert torch.equal(real, clip)
    assert sum(w.valid_frames for w in windows) == length
    for window in windows:
        assert window.frames.shape == (16, 3, 2, 2)
        assert window.frame_indices == list(range(7 + window.offset, 7 + window.offset + window.valid_frames))
        if window.valid_frames < 16:
            assert torch.equal(window.frames[window.valid_frames:], window.frames[-1:].expand(16-window.valid_frames, -1, -1, -1))


def test_empty_timeline_is_rejected():
    with pytest.raises(ValueError):
        list(temporal_windows(torch.empty(0, 3, 2, 2)))
