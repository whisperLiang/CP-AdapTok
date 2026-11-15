import json

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
