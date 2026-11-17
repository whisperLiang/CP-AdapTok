import copy
import json
from argparse import Namespace
from dataclasses import replace

import cv2
import numpy as np
import pytest
import torch
import yaml

from adaptok_hr.runtime import file_hash, stable_hash, write_json
from adaptok_hr.tests.test_training import settings


@pytest.fixture
def native_record(tmp_path):
    path = tmp_path / "source"
    path.mkdir()
    array = np.full((1080, 1920, 3), 100, np.uint8)
    hashes = []
    for i in range(4):
        target = path / f"{i:05d}.png"
        cv2.imwrite(str(target), array + i)
        hashes.append({"file": target.name, "sha256": file_hash(target)})
    return {"path": str(path), "content_id": "unit:source", "source": "generated unit-test pixels",
            "resolution": [1920, 1080], "original_frames": 4, "fps": 30, "resized": False,
            "role": "train", "official_split": "train", "sha256": stable_hash(hashes), "frame_hashes": hashes}


def test_real_trainer_lifecycle_resume_and_seal(small_config, native_record, tmp_path, monkeypatch):
    from adaptok_hr.training.train import exclusive_run, run_training

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    index = {"splits": {"train": [native_record], "validation": [], "test": []}}
    index["identity"] = stable_hash(index)
    manifest = tmp_path / "index.json"
    write_json(manifest, index)
    train = {**settings(), "seed": 9, "sequence_schedule": [[0, 2]], "checkpoint_every": 1,
             "validation_every": 100}
    config = {"model": small_config.to_dict(), "training": train,
              "data": {"index": str(manifest)}, "evaluation": {"validation_ids": [], "amp_bf16": False}}
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    output = tmp_path / "run"
    args = Namespace(config=str(config_path), output=str(output), steps=2, resume=None,
                     device="cpu", checkpoints=str(tmp_path / "checkpoints"),
                     overfit="unit:source", overfit_frames=2)
    with exclusive_run(output):
        with pytest.raises(ValueError, match="writer"):
            with exclusive_run(output):
                pass
        run_training(args)
    assert json.loads((output / "completion.json").read_text())["training_frames"] == 4
    with pytest.raises(ValueError, match="resume"):
        run_training(args)
    args.resume = str(tmp_path / "checkpoints/run/step_000002.pt")
    args.steps = 4
    run_training(args)
    rows = [json.loads(line) for line in (output / "training.jsonl").read_text().splitlines()]
    assert [r["step"] for r in rows] == [1, 2, 3, 4]
    assert rows[-1]["training_frames"] == 8
    write_json(output / "final_test_seal.json", {"frozen": True})
    with pytest.raises(ValueError, match="sealed"):
        run_training(args)


def test_evaluator_reads_only_packets_and_saves_actual_bytes(small_model, native_record, tmp_path):
    from adaptok_hr.evaluation.evaluate import evaluate_source, measured_quality, verify_endpoints

    summary, rows = evaluate_source(small_model.eval(), native_record, 8, 3, amp_bf16=False,
                                    output=tmp_path / "evaluation")
    assert summary["frames"] == 3 and summary["at_least_64_contiguous_frames"] is False
    packets = sorted((tmp_path / "evaluation").glob("*.bin"))
    assert len(packets) == 3 and sum(p.stat().st_size for p in packets) == summary["bytes"]
    assert rows[0]["frame_type"] == "I" and rows[1]["frame_type"] == "P"
    assert summary["input"]["engineering_crop"] is True
    with pytest.raises(ValueError, match="1080p"):
        measured_quality(torch.zeros(1, 3, 64, 64), torch.zeros(1, 3, 64, 64), lambda a, b: {})
    state = Namespace(rgb=torch.zeros(1), feature=torch.zeros(1))
    with pytest.raises(RuntimeError, match="diverged"):
        verify_endpoints(state, Namespace(rgb=torch.ones(1), feature=state.feature))


def test_index_build_and_external_license_identity(native_record, tmp_path):
    from adaptok_hr.data.manifest import build_index, load_index

    for split in ("train", "validation", "test"):
        rows = [native_record] if split == "train" else []
        write_json(tmp_path / f"cpat_{split}.json", {"records": rows})
    extra = copy.deepcopy(native_record)
    extra.update(path=str(tmp_path / "external-source"), content_id="unit:external", sha256="2" * 64)
    external = tmp_path / "external.json"
    write_json(external, {"license": "unit-test license", "usage_conditions": "unit testing", "records": [extra]})
    index = build_index(tmp_path, [external])
    assert index["summary"]["train"]["sources"] == 2
    path = tmp_path / "index.json"
    write_json(path, index)
    assert load_index(path)["identity"] == index["identity"]
    index["color_format"] = "changed"
    write_json(path, index)
    with pytest.raises(ValueError, match="identity"):
        load_index(path)


@pytest.mark.parametrize("changes", [
    {"height": 32}, {"width": 0}, {"gop": 0}, {"channels": (12,)}, {"heads": 5},
    {"channels": (0,) * 7}, {"codebook_size": 100}, {"embedding_dim": 8},
    {"budgets": ()}, {"budgets": (8, 4)}, {"budgets": (0, 8)}, {"budgets": (4, 8192)},
    {"attention_blocks": -1},
])
def test_invalid_config_rejected_at_boundary(small_config, changes):
    with pytest.raises(ValueError):
        replace(small_config, **changes)


def test_scorer_readiness_and_checkpoint_resume(small_model, tmp_path):
    from adaptok_hr.data.dataset import ClipSampler
    from adaptok_hr.models.scorer import BudgetScorer
    from adaptok_hr.training.train_scorer import restore_scorer, save_scorer, train_scorer

    readiness = tmp_path / "not_ready.json"
    write_json(readiness, {"scorer_ready": False})
    with pytest.raises(ValueError, match="readiness"):
        train_scorer(Namespace(readiness=readiness))
    scorer = BudgetScorer(16, small_model.config.budgets)
    optimizer = torch.optim.AdamW(scorer.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.)
    sampler = ClipSampler([{"original_frames": 64}], 4)
    path = tmp_path / "scorer.pt"
    identity = {"frozen": "hash"}
    save_scorer(path, scorer, optimizer, scheduler, sampler, 3, identity)
    assert restore_scorer(path, scorer, optimizer, scheduler, sampler, identity) == 3
    with pytest.raises(ValueError, match="identity"):
        restore_scorer(path, scorer, optimizer, scheduler, sampler, {})


def test_scorer_runner_freezes_checkpoint_and_can_resume(small_model, native_record, tmp_path, monkeypatch):
    from adaptok_hr.data.dataset import ClipSampler
    from adaptok_hr.training.checkpoint import save_training
    from adaptok_hr.training.train import JointTrainer
    from adaptok_hr.training.train_scorer import train_scorer

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = {"model": small_model.config.to_dict(), "training": settings()}
    trainer = JointTrainer(small_model, config["training"])
    sampler = ClipSampler([native_record], 1)
    checkpoint = tmp_path / "tokenizer.pt"
    save_training(checkpoint, trainer, sampler, config, "unit-manifest", 1, 2)
    before = file_hash(checkpoint)
    index = {"splits": {"train": [native_record], "validation": [], "test": []}}
    index["identity"] = stable_hash(index)
    write_json(tmp_path / "index.json", index)
    write_json(tmp_path / "ready.json", {"scorer_ready": True, "scope": "unit test only"})
    args = Namespace(readiness=tmp_path / "ready.json", checkpoint=checkpoint, index=tmp_path / "index.json",
                     device="cpu", output=tmp_path / "scorer", resume=None, steps=2)
    args.output.mkdir()
    train_scorer(args)
    args.resume, args.steps = args.output / "step_000002.pt", 3
    train_scorer(args)
    assert file_hash(checkpoint) == before
    done = json.loads((args.output / "completion.json").read_text())
    assert done["completed_steps"] == 3 and done["tokenizer_unchanged"]


def test_frame_and_source_boundaries_are_checked(small_model, native_record):
    from adaptok_hr.data.manifest import validate_row

    for frame in (torch.zeros(1, 3, 64, 64), torch.zeros(1, 3, 128, 256, dtype=torch.long),
                  torch.full((1, 3, 128, 256), float("nan")), torch.full((1, 3, 128, 256), 2.)):
        with pytest.raises(ValueError):
            small_model.encode_features(frame)
    for changes, split in (({"sha256": ""}, "train"), ({"role": "test"}, "train"),
                           ({"content_id": "DAVIS2017:cows"}, "train")):
        with pytest.raises(ValueError):
            validate_row({**native_record, **changes}, split)
