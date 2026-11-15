import pytest
import torch

from cp_adaptok.data import VideoSpec, sample_training_spec


def test_training_visits_reproducible_windows_inside_declared_source_span():
    source = {"path": "clip.mp4", "frames": 64, "start": 10, "fps": 30}
    generator = torch.Generator().manual_seed(47)
    first = [sample_training_spec(source, 8, generator) for _ in range(12)]
    restored = torch.Generator().manual_seed(47)
    assert first == [sample_training_spec(source, 8, restored) for _ in range(12)]
    assert len({spec.start for spec in first}) > 1
    assert all(10 <= spec.start and spec.start + spec.frames <= 74 for spec in first)
    assert all(spec.frames == 8 and spec.path == source["path"] for spec in first)
    assert source["frames"] == 64 and source["start"] == 10


def test_full_length_window_preserves_start_and_can_resume_sampling_rng():
    source = VideoSpec("clip.mp4", frames=16, start=7)
    generator = torch.Generator().manual_seed(20)
    assert sample_training_spec(source, 16, generator) == source
    state = generator.get_state()
    expected = sample_training_spec(source, 4, generator)
    restored = torch.Generator()
    restored.set_state(state)
    assert sample_training_spec(source, 4, restored) == expected


@pytest.mark.parametrize("length", [0, -1, 17, 4.0, True])
def test_invalid_training_window_is_rejected_without_consuming_rng(length):
    generator = torch.Generator().manual_seed(3)
    state = generator.get_state()
    with pytest.raises(ValueError):
        sample_training_spec(VideoSpec("clip.mp4", frames=16), length, generator)
    assert torch.equal(state, generator.get_state())
