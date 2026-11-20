from copy import deepcopy

import pytest
import torch

from aip_vtok.c1_followup.io import tensor_digest
from aip_vtok.joint.checkpoint import CheckpointStore, Journal, make_checkpoint, restore_training
from aip_vtok.joint.protocol import CONFIG, open_experiment
from aip_vtok.joint.trainer import Trainer, Window


def test_journal_replays_exact_optimizer_boundary_and_discards_uncommitted_tail(tmp_path):
    path = tmp_path / "updates.jsonl"
    journal = Journal(path, None)
    journal.append({"step": 1, "window": {"start": 16}})
    marker = journal.marker()
    journal.append({"step": 2, "window": {"start": 17}})
    resume = Journal(path, marker)
    assert resume.marker() == marker
    assert "17" not in path.read_text()
    with pytest.raises(ValueError, match="out-of-order"):
        resume.append({"step": 1})
    path.write_text(path.read_text().replace("16", "15"))
    with pytest.raises(ValueError, match="differs"):
        Journal(path, marker)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA required")
def test_cuda_checkpoint_replays_identical_model_and_optimizer(tmp_path):
    run = open_experiment(CONFIG, 1)
    trainer = Trainer(run.codec, run.config, 1)
    rows = run.data.catalog("train")
    window = Window("DAVIS2017:bear", 0, 1)
    clip = run.data.read("train", window.video, 0, 1)
    journal = Journal(tmp_path / "updates.jsonl", None)
    journal.append(trainer.update(clip, window, 1))
    store = CheckpointStore(tmp_path / "checkpoints", run.identity)
    store.write(make_checkpoint(trainer, rows, journal, run.identity, {}), journal)
    artifact, _ = store.load()
    expected = trainer.update(clip, window, 2)
    digest = tensor_digest(run.codec)
    optimizer = deepcopy(trainer.optimizer.state_dict())
    corrupt = artifact | {"step": 2}
    with pytest.raises(ValueError, match="optimizer step"):
        restore_training(trainer, rows, corrupt)
    corrupt = artifact | {"dataset_sampler": {"cursor": 1, "next_window": {"video": "other"}}}
    with pytest.raises(ValueError, match="data window"):
        restore_training(trainer, rows, corrupt)
    restore_training(trainer, rows, artifact)
    assert trainer.update(clip, window, 2) == expected
    assert tensor_digest(run.codec) == digest
    after = trainer.optimizer.state_dict()
    assert optimizer["param_groups"] == after["param_groups"]
    for key, tensors in optimizer["state"].items():
        assert all(torch.equal(value, after["state"][key][name]) for name, value in tensors.items())
    with (store.directory / "step_00001.pt").open("ab") as handle:
        handle.write(b"tainted")
    with pytest.raises(ValueError, match="SHA256"):
        store.load()
