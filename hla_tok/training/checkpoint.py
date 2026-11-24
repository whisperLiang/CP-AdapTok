"""Atomic append-only artifacts for reproducible HLA-Tok training."""

import hashlib
import os
import random
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

SCHEMA = 1


@dataclass(frozen=True)
class ResumeIdentity:
    stage: str
    config_sha256: str
    dataset_sha256: str
    git_commit: str
    implementation_sha256: str = ""


@dataclass(frozen=True)
class Checkpoint:
    state: dict

    def restore(self, model, optimizer, scheduler, budget_rng) -> dict:
        model.load_state_dict(self.state["model"])
        optimizer.load_state_dict(self.state["optimizer"])
        scheduler.load_state_dict(self.state["scheduler"])
        torch.set_rng_state(self.state["rng"]["torch"])
        random.setstate(self.state["rng"]["python"])
        cuda = self.state["rng"]["cuda"]
        if cuda is not None:
            torch.cuda.set_rng_state(cuda)
        budget_rng.set_state(self.state["budget_rng"])
        return self.state


def implementation_hash(root: Path) -> str:
    digest = hashlib.sha256()
    sources = [*root.joinpath("hla_tok").rglob("*.py"), root / "scripts/hla_tok/run_phase1.py"]
    for source in sorted(sources):
        digest.update(str(source.relative_to(root)).encode() + b"\0")
        digest.update(hashlib.sha256(source.read_bytes()).digest())
    return digest.hexdigest()


def _compatible(stored: dict, identity: ResumeIdentity) -> bool:
    proposed = asdict(identity)
    if identity.implementation_sha256:
        proposed["git_commit"] = stored["git_commit"]
    return stored == proposed


def read_checkpoint(path: Path, identity: ResumeIdentity | None = None) -> Checkpoint:
    state = torch.load(path, map_location="cpu", weights_only=True)
    required = {"schema", "identity", "model", "optimizer", "scheduler", "rng", "budget_rng",
                "step", "data_cursor", "counters", "amp"}
    if state.keys() != required or state["schema"] != SCHEMA:
        raise ValueError("incompatible checkpoint schema")
    if identity is not None and not _compatible(state["identity"], identity):
        raise ValueError("checkpoint identity differs from the frozen training run")
    if state["step"] != state["data_cursor"] or state["step"] < 1:
        raise ValueError("checkpoint step and deterministic data cursor disagree")
    return Checkpoint(state)


def save_checkpoint(path, identity, model, optimizer, scheduler, budget_rng, *, step, data_cursor, counters):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    rng = {"torch": torch.get_rng_state(), "python": random.getstate(),
           "cuda": torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None}
    state = {"schema": SCHEMA, "identity": asdict(identity), "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(), "amp": None,
             "rng": rng, "budget_rng": budget_rng.get_state(), "step": step,
             "data_cursor": data_cursor, "counters": counters}
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".hla-", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            torch.save(state, stream)
            stream.flush()
            os.fsync(stream.fileno())
            reloaded = read_checkpoint(temporary, identity).state
            if reloaded["step"] != step or any(
                not torch.equal(tensor.detach().cpu(), reloaded["model"][name])
                for name, tensor in state["model"].items()
            ):
                raise RuntimeError("checkpoint verification failed after writing")
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
