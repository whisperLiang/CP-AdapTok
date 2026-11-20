"""Hash-verified checkpoints, optimizer/RNG replay, and a recoverable update journal."""

import hashlib
import json
import os
import random
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from aip_vtok.diagnostics.diag0.io import json_at
from aip_vtok.runtime import file_hash, save_checkpoint, write_json

from .scan import SLOT_COUNT, scan_digest

SCHEMA = 1
FROZEN = ("backbone.", "projector.")


def rng_state() -> dict:
    name, data, position, gaussian, cached = np.random.get_state()
    return {"python": random.getstate(), "numpy": (name, data.tolist(), position, gaussian, cached),
            "torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all()}


def restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    name, data, position, gaussian, cached = state["numpy"]
    np.random.set_state((name, np.array(data, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state_all(state["cuda"])


@contextmanager
def isolated_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def model_state(codec) -> dict:
    return {name: tensor for name, tensor in codec.state_dict().items() if not name.startswith(FROZEN)}


def load_model(codec, artifact: dict) -> None:
    model = artifact["model"]
    if model_state(codec).keys() != model.keys():
        raise ValueError("checkpoint keys disagree with the Joint model")
    codec.load_state_dict(model, strict=False)
    codec.scan.frozen = artifact["scan"]["frozen"]
    if codec.scan.frozen:
        slots = codec.scan.locked_order
        if not torch.equal(slots.sort().values, torch.arange(SLOT_COUNT, device=slots.device)):
            raise ValueError("checkpoint scan contains invalid or duplicate slots")
        if scan_digest(slots) != artifact["scan"]["scan_sha256"]:
            raise ValueError("checkpoint frozen scan SHA256 differs")
        codec.scan.score.requires_grad_(False)
    if codec.scan.record() != artifact["scan"]:
        raise ValueError("checkpoint scan identity differs")


class Journal:
    def __init__(self, path: Path, marker: dict | None):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        self.hash = hashlib.sha256()
        self.step = 0
        self.offset = 0
        if marker is None:
            path.touch(exist_ok=False)
        else:
            self._restore(marker)

    def _restore(self, marker: dict) -> None:
        with self.path.open("r+b") as handle:
            while handle.tell() < marker["offset"]:
                line = handle.readline()
                record = json.loads(line)
                self.step += 1
                if record["step"] != self.step:
                    raise ValueError("training journal has a missing or duplicate update")
                self.hash.update(line)
            self.offset = handle.tell()
            if self.marker() != marker:
                raise ValueError("training journal differs from checkpoint")
            handle.truncate(self.offset)

    def append(self, record: dict) -> None:
        if record["step"] != self.step + 1:
            raise ValueError("training journal received an out-of-order update")
        line = (json.dumps(record, allow_nan=False, separators=(",", ":")) + "\n").encode()
        with self.path.open("ab") as handle:
            handle.write(line)
        self.hash.update(line)
        self.step += 1
        self.offset += len(line)

    def marker(self) -> dict:
        return {"step": self.step, "offset": self.offset, "sha256": self.hash.hexdigest()}

    def sync(self) -> None:
        with self.path.open("rb") as handle:
            os.fsync(handle.fileno())


def dataset_cursor(trainer, rows: dict, step: int) -> dict:
    next_window = trainer.next_window(rows, step + 1)
    return {"cursor": step, "next_window": None if next_window is None else asdict(next_window)}


def make_checkpoint(trainer, rows: dict, journal: Journal, identity: dict, progress: dict) -> dict:
    step = journal.step
    if trainer.scheduler.last_epoch != step:
        raise ValueError("cannot checkpoint inconsistent optimizer and journal counters")
    return {"schema": SCHEMA, "step": step, "identity": identity, "model": model_state(trainer.codec),
            "optimizer": trainer.optimizer.state_dict(), "scheduler": trainer.scheduler.state_dict(),
            "amp_scaler": None, "rng": rng_state(), "dataset_sampler": dataset_cursor(trainer, rows, step),
            "budget_sampler": trainer.budgets.state_dict(), "scan": trainer.codec.scan.record(),
            "journal": journal.marker(), "progress": progress}


def restore_training(trainer, rows: dict, artifact: dict) -> None:
    step = artifact["step"]
    frozen = step >= trainer.config["scan_freeze_step"]
    slots = artifact["optimizer"]["state"]
    latest_update = max((int(v["step"]) for v in slots.values()), default=0)
    if (artifact["dataset_sampler"] != dataset_cursor(trainer, rows, step)
            or artifact["journal"]["step"] != step or artifact["scheduler"]["last_epoch"] != step
            or latest_update != step or artifact["scan"]["frozen"] != frozen):
        raise ValueError("checkpoint optimizer step, scan state, or data window differs")
    load_model(trainer.codec, artifact)
    trainer.optimizer.load_state_dict(artifact["optimizer"])
    trainer.scheduler.load_state_dict(artifact["scheduler"])
    trainer.budgets.load_state_dict(artifact["budget_sampler"])
    restore_rng(artifact["rng"])


class CheckpointStore:
    def __init__(self, directory: Path, identity: dict):
        self.directory = directory
        self.index = directory / "index.json"
        self.identity = identity

    def entries(self) -> dict:
        if not self.index.exists():
            return {}
        index = json_at(self.index)
        if index["identity"] != self.identity:
            raise ValueError("checkpoint index identity differs from source, config, seed, or assets")
        return index["checkpoints"]

    def write(self, artifact: dict, journal: Journal) -> dict:
        entries = self.entries()
        step = artifact["step"]
        if str(step) in entries:
            raise FileExistsError(f"checkpoint step {step} already exists")
        journal.sync()
        path = self.directory / f"step_{step:05d}.pt"
        save_checkpoint(path, artifact)
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
        record = {"step": step, "file": path.name, "sha256": file_hash(path), "scan": artifact["scan"]}
        entries[str(step)] = record
        write_json(self.index, {"identity": self.identity, "checkpoints": entries})
        return record

    def load(self, step: int | None = None) -> tuple[dict, dict]:
        entries = self.entries()
        if not entries:
            raise FileNotFoundError(f"no verified checkpoint at {self.directory}")
        step = max(map(int, entries)) if step is None else step
        record = entries[str(step)]
        path = self.directory / f"step_{step:05d}.pt"
        if path.name != record["file"] or file_hash(path) != record["sha256"]:
            raise ValueError(f"checkpoint SHA256 or filename mismatch: {path}")
        artifact = torch.load(path, weights_only=True, map_location="cpu")
        if (artifact["identity"] != self.identity or artifact["schema"] != SCHEMA or artifact["step"] != step
                or artifact["scan"] != record["scan"] or record["step"] != step):
            raise ValueError(f"checkpoint identity, model version, or optimizer step mismatch: {path}")
        return artifact, record
