"""Strict checkpoint loading, explicit grid migration and optimizer-step snapshots."""

import copy
import hashlib
import random
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from cp_adaptok.models.causal_progressive_codec import CausalProgressiveCodec, CodecConfig
from cp_adaptok.runtime import load_checkpoint, metadata, resolve
from experiments.fixed2048.codec import Fixed2048Reconstructor

FORMAT = "fixed2048_v1"


def experiment_source_sha256():
    directory = Path(__file__).parent
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.py")):
        digest.update(str(path.relative_to(directory)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def sha256(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require_checkpoint(path):
    path = resolve(path)
    if not path.is_file():
        raise FileNotFoundError(f"required trained checkpoint is missing: {path}")
    return path


def migrate_grid(state, source, target):
    if source.grid != target.grid and source.kv_grid != target.kv_grid:
        raise ValueError("decoder and encoder grid ablations must be separate")
    if replace(target, grid=source.grid, kv_grid=source.kv_grid) != source:
        raise ValueError("grid migration cannot change other model settings")
    converted, ledger = {}, []
    for name, value in state.items():
        operation = "reuse"
        if name == "decoder.projector.query" and source.grid != target.grid:
            grid = value.transpose(1, 2).reshape(1, source.dim, *source.grid)
            value = F.interpolate(grid, size=target.grid, mode="bilinear", align_corners=False)
            value = value.flatten(2).transpose(1, 2).contiguous()
            operation = "bilinear_2d"
        converted[name] = value
        ledger.append(
            {
                "parameter": name,
                "operation": operation,
                "source_shape": list(state[name].shape),
                "target_shape": list(value.shape),
            }
        )
    return converted, ledger


def load_reconstructor(path, device="cuda:0", decoder_grid=None, encoder_kv_grid=None, profile=False):
    path = require_checkpoint(path)
    artifact = torch.load(path, map_location="cpu", weights_only=True)
    fixed = artifact.get("experiment_format") == FORMAT
    if not fixed:
        artifact = load_checkpoint(path)
    if artifact["status"] != "trained":
        raise ValueError("experiment requires an actual trained checkpoint")
    source_config = CodecConfig(**artifact["model_config"])
    target_config = replace(
        source_config,
        grid=tuple(decoder_grid or source_config.grid),
        kv_grid=tuple(encoder_kv_grid or source_config.kv_grid),
    )
    state, ledger = migrate_grid(artifact["model"], source_config, target_config)
    source = CausalProgressiveCodec(target_config)
    digest = sha256(path)
    if not fixed:
        source.load_state_dict(state, strict=True)
    model = Fixed2048Reconstructor(source, identity=digest[:32], profile=profile)
    if fixed:
        model.load_state_dict(state, strict=True)
    identity = {
        "path": str(path),
        "sha256": digest,
        "model_config": asdict(target_config),
        "source_model_config": asdict(source_config),
        "checkpoint_step": artifact["step"],
        "training_status": artifact["status"],
        "strict_load": True,
        "migration": ledger,
        "source_metadata": artifact.get("metadata"),
        "experiment_source": artifact.get("source_identity"),
        "trained_path": artifact["config"].get("training", {}).get("path", "historical_VQ"),
        "experiment_source_sha256": experiment_source_sha256(),
    }
    model.provenance = identity
    return model.to(device).eval(), identity


def rng_state(generator):
    numpy = np.random.get_state()
    return {
        "sampling": generator.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "python": random.getstate(),
        "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
    }


def restore_rng(state, generator):
    generator.set_state(state["sampling"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        torch.cuda.set_rng_state_all(state["cuda"])
    random.setstate(state["python"])
    numpy = state["numpy"]
    np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32), *numpy[2:]))


def save_training(path, model, optimizer, scheduler, generator, step, config, history, source_identity):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    details = metadata(config, "trained")
    details["experiment_source_sha256"] = experiment_source_sha256()
    torch.save(
        {
            "experiment_format": FORMAT,
            "model_config": asdict(model.config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "rng": rng_state(generator),
            "step": step,
            "config": config,
            "history": history,
            "status": "trained",
            "source_identity": source_identity,
            "metadata": details,
        },
        temporary,
    )
    temporary.replace(path)


def resume_training(path, model, optimizer, scheduler, generator, config):
    artifact = torch.load(require_checkpoint(path), map_location="cpu", weights_only=True)
    if artifact.get("experiment_format") != FORMAT or artifact["config"] != config:
        raise ValueError("resume requires the identical declared experiment configuration")
    if artifact["model_config"] != asdict(model.config):
        raise ValueError("resume architecture mismatch")
    model.load_state_dict(artifact["model"], strict=True)
    optimizer.load_state_dict(artifact["optimizer"])
    scheduler.load_state_dict(artifact["scheduler"])
    restore_rng(artifact["rng"], generator)
    return artifact["step"], copy.deepcopy(artifact["history"]), artifact["source_identity"]


def official_backbone_identity(model):
    from third_party.DCVC.src.utils.common import get_state_dict

    result = {}
    for name, file in (("intra", "cvpr2026_image.pth.tar"), ("ld", "cvpr2026_video_ld.pth.tar")):
        path = require_checkpoint(f"checkpoints/{file}")
        weights = get_state_dict(path)
        actual = getattr(model, name).model.state_dict()
        same = set(weights) == set(actual) and all(torch.equal(weights[k], actual[k].cpu()) for k in weights)
        result[name] = {"path": str(path), "sha256": sha256(path), "matches_loaded_backbone": same}
    return result
