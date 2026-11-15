import json
from dataclasses import replace

import pytest
import torch

from cp_adaptok.runtime import MissingPrerequisites, load_checkpoint, preflight, save_checkpoint


def test_missing_weights_are_explicit_and_machine_readable(tmp_path):
    config = {
        "paths": {"ld_checkpoint": str(tmp_path / "ld.pt"), "intra_checkpoint": str(tmp_path / "i.pt")},
        "data": {},
    }
    with pytest.raises(MissingPrerequisites) as error:
        preflight(config, "train", False)
    assert "ld_checkpoint" in str(error.value) and "intra_checkpoint" in str(error.value)
    assert len(error.value.conditions) >= 2
    for purpose, loss in [("baseline", {}), ("train", {"latent": 1.0})]:
        with pytest.raises(MissingPrerequisites) as unavailable:
            preflight({**config, "loss": loss}, purpose, False)
        assert any("ld_checkpoint" in item for item in unavailable.value.conditions)


def test_checkpoint_restores_model_optimizer_and_sampling_rng(codec, tmp_path):
    optimizer = torch.optim.Adam([codec.tokenizer.queries], lr=0.001)
    generator = torch.Generator().manual_seed(81)
    path = tmp_path / "resume.pt"
    save_checkpoint(path, codec, optimizer, generator, step=3, config={"stage": 1}, status="smoke")
    expected = torch.rand(4, generator=generator)
    loaded = load_checkpoint(path)
    generator.set_state(loaded["sampling_rng"])
    assert torch.equal(torch.rand(4, generator=generator), expected)
    assert loaded["step"] == 3 and loaded["status"] == "smoke"
    assert json.loads(json.dumps(loaded["config"])) == {"stage": 1}


def test_invalid_model_frame_and_shape_fail_before_allocating_reference(codec):
    state = codec.empty_state()
    with pytest.raises(ValueError, match="model_id"):
        codec.begin_frame(replace(state, model_id="ff" * 16), 0, "I", 16, 24)
    with pytest.raises(ValueError, match="frame type/id"):
        codec.begin_frame(state, -1, "I", 16, 24)
    with pytest.raises(ValueError, match="shape/QP"):
        codec.begin_frame(state, 0, "I", 0, 24)
    assert state.frame_id == -1 and state.feature is None
