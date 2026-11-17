import copy

import pytest
import torch

from experiments.fixed2048.checkpoint import resume_training, save_training
from experiments.fixed2048.tests.test_codec import make_model
from experiments.fixed2048.train import train_update


def update(model, optimizer):
    rgb = torch.rand(2, 3, 64, 64)
    return train_update(model, rgb, optimizer, {"tbptt": 2, "amp_bf16": False, "clip_grad_norm": 1.0})


def test_optimizer_resume_continues_next_update_exactly(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(12)
    model = make_model().configure_training("R1").train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    generator = torch.Generator().manual_seed(9)
    first = update(model, optimizer)
    scheduler.step()
    checkpoint = tmp_path / "resume.pt"
    save_training(checkpoint, model, optimizer, scheduler, generator, 1, {"steps": 2}, {"train": [first]}, {})
    expected_row = update(model, optimizer)
    scheduler.step()
    expected = copy.deepcopy(model.state_dict())
    completed, history, _ = resume_training(checkpoint, model, optimizer, scheduler, generator, {"steps": 2})
    assert completed == 1 and len(history["train"]) == 1
    actual_row = update(model, optimizer)
    scheduler.step()
    for name, value in model.state_dict().items():
        assert torch.equal(value, expected[name]), name
    assert actual_row["mse"] == expected_row["mse"]
    assert actual_row["module_gradient_norm"]["tokenizer"] > 0
    assert actual_row["module_update_norm"]["decoder"] > 0
    assert actual_row["frozen_backbone_gradients"] == 0
    with pytest.raises(ValueError, match="identical"):
        resume_training(checkpoint, model, optimizer, scheduler, generator, {"steps": 3})


def test_training_writer_saves_step0_and_reconciles_uncommitted_steps(tmp_path, monkeypatch):
    import json

    import experiments.fixed2048.train as training
    from cp_adaptok.runtime import read_config

    torch.set_num_threads(2)
    config = read_config("configs/fixed2048.yaml")
    config["device"] = "cpu"
    config["test_seal"] = str(tmp_path / "unsealed.json")
    config["training"].update(steps=2, save_every=1, validate_every=1, sequence_length=2)
    row = {"content_id": "train:tiny", "path": "tiny", "resolution": [1920, 1080], "original_frames": 4}
    monkeypatch.setattr(training, "load_reconstructor", lambda *a, **k: (make_model(), {}))
    monkeypatch.setattr(training, "records", lambda *a, **k: ([row], {"train": "frozen"}))
    monkeypatch.setattr(training, "read_clip", lambda *a, **k: torch.full((2, 3, 64, 64), 0.5))
    monkeypatch.setattr(training, "save_image", lambda *a, **k: None)
    results, checkpoints = tmp_path / "results", tmp_path / "checkpoints"
    training.run_training(config, results, checkpoints, stop_after=1)
    assert (checkpoints / "step0.pt").exists()
    assert (checkpoints / "best_validation.pt").exists()
    with (results / "train.jsonl").open("a") as log:
        log.write(json.dumps({"step": 999, "mse": 0}) + "\n")
    history = training.run_training(config, results, checkpoints, resume=checkpoints / "latest.pt")
    assert [row["step"] for row in history["train"]] == [1, 2]
    logged = [json.loads(line) for line in (results / "train.jsonl").read_text().splitlines()]
    assert [row["step"] for row in logged] == [1, 2]
    completion = json.loads((results / "completion.json").read_text())
    assert completion["actual_optimizer_steps"] == 2
    assert completion["actual_training_frames"] == 4
    with pytest.raises(FileExistsError):
        training.run_training(config, results, checkpoints)
