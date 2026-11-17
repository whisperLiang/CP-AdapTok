"""Atomic optimizer-boundary checkpoints and log rollback to the committed boundary."""

import json
import random
from pathlib import Path

import numpy as np
import torch

from ..codec.codec import tensor_digest
from ..config import ModelConfig
from ..models.model import AdapTokHR
from ..runtime import file_hash, source_identity, stable_hash, write_json


def rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def save_training(path, trainer, sampler, config, manifest_identity, completed_steps, training_frames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {"version": 1, "model": trainer.model.state_dict(), "optimizer": trainer.optimizer.state_dict(),
             "scheduler": trainer.scheduler.state_dict(), "rng": rng_state(), "sampling": sampler.state_dict(),
             "completed_steps": completed_steps, "training_frames": training_frames, "config": config,
             "config_identity": stable_hash(config), "manifest_identity": manifest_identity}
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    metadata = {k: state[k] for k in ("version", "completed_steps", "training_frames", "config",
                                     "config_identity", "manifest_identity")}
    metadata.update(checkpoint_sha256=file_hash(path), model_sha256=tensor_digest(state["model"]).hex(),
                    codebook_sha256=tensor_digest({"codebook": trainer.model.vq.codebook.weight}).hex(),
                    source_sha256=source_identity(),
                    optimizer_saved=True, scheduler_saved=True, rng_saved=True, sampling_saved=True)
    write_json(path.with_suffix(".json"), metadata)
    return metadata


def restore_training(path, trainer, sampler, config, manifest_identity):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["version"] != 1 or state["config_identity"] != stable_hash(config):
        raise ValueError("checkpoint configuration identity mismatch")
    if state["manifest_identity"] != manifest_identity:
        raise ValueError("checkpoint manifest/source identity mismatch")
    trainer.model.load_state_dict(state["model"], strict=True)
    trainer.optimizer.load_state_dict(state["optimizer"])
    trainer.scheduler.load_state_dict(state["scheduler"])
    sampler.load_state_dict(state["sampling"])
    restore_rng(state["rng"])
    return {"completed_steps": state["completed_steps"], "training_frames": state["training_frames"]}


def reconcile_log(path, completed):
    path = Path(path)
    if not path.exists():
        if completed:
            raise ValueError("committed checkpoint has no training log")
        return 0
    rows = []
    lines = path.read_text().splitlines()
    for index, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError("corrupt training log before final partial line") from None
    kept = [r for r in rows if r["step"] <= completed]
    if [r["step"] for r in kept] != list(range(1, completed + 1)):
        raise ValueError("committed optimizer sequence disagrees with log")
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in kept))
    temporary.replace(path)
    return len(rows) - len(kept)


def load_model(path, device="cuda"):
    state = torch.load(path, map_location="cpu", weights_only=False)
    model = AdapTokHR(ModelConfig(**state["config"]["model"]))
    model.load_state_dict(state["model"], strict=True)
    return model.to(device).eval(), state
