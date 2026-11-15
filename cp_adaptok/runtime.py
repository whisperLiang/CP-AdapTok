import hashlib
import importlib.util
import json
import os
import random
import subprocess
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml

from cp_adaptok.models.causal_progressive_codec import CausalProgressiveCodec, CodecConfig

ROOT = Path(__file__).resolve().parents[1]


class MissingPrerequisites(RuntimeError):
    def __init__(self, conditions):
        self.conditions = conditions
        super().__init__("Missing prerequisites: " + "; ".join(conditions))


def resolve(path):
    path = Path(path).expanduser()
    return path if path.is_absolute() else ROOT / path


def read_config(path):
    path = resolve(path)
    with path.open() as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be a YAML mapping")
    parent = config.pop("extends", None)
    return merge_config(read_config(path.parent / parent), config) if parent else config


def merge_config(base, overrides):
    result = dict(base)
    for key, value in overrides.items():
        result[key] = (
            merge_config(result[key], value)
            if isinstance(value, dict) and isinstance(result.get(key), dict)
            else value
        )
    return result


def seed_everything(seed, device="cpu", memory_fraction=0.18):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    if str(device).startswith("cuda"):
        if not torch.cuda.is_available():
            raise MissingPrerequisites(["CUDA device unavailable"])
        torch.cuda.manual_seed_all(seed)
        torch.cuda.set_per_process_memory_fraction(memory_fraction, torch.device(device))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False


def _missing_files(paths, names):
    return [
        f"{name}={paths.get(name) or '<not configured>'}"
        for name in names
        if not paths.get(name) or not resolve(paths[name]).is_file()
    ]


def _model_prerequisites(config, purpose, allow_untrained):
    paths = config.get("paths", {})
    artifact = paths.get("checkpoint")
    artifact_present = bool(artifact and resolve(artifact).is_file())
    missing = []
    if not artifact_present and not allow_untrained:
        missing += _missing_files(paths, ["ld_checkpoint", "intra_checkpoint"])
        if purpose != "train" and purpose != "baseline":
            missing += _missing_files(paths, ["checkpoint"])
    return missing


def _native_prerequisites(config, purpose):
    native_teacher = purpose == "train" and config.get("loss", {}).get("latent", 0) > 0
    if purpose != "baseline" and not native_teacher:
        return []
    missing = _missing_files(config.get("paths", {}), ["ld_checkpoint", "intra_checkpoint"])
    extensions = ("MLCodec_extensions_cpp", "inference_extensions_cuda")
    missing += [
        f"native codec/teacher needs official {name}; matching CUDA/PyTorch and nvcc"
        for name in extensions
        if importlib.util.find_spec(name) is None
    ]
    if native_teacher and not missing:
        from cp_adaptok.backbones.native_baseline import native_latent_api_available

        if not native_latent_api_available():
            missing.append("native teacher needs decoded-latent API; run bash scripts/setup_native.sh")
    return missing


def _data_prerequisites(config):
    missing = []
    for split, specs in config.get("data", {}).items():
        for spec in specs:
            if not resolve(spec["path"]).exists():
                missing.append(f"{split} data={spec['path']}")
    return missing


def preflight(config, purpose, allow_untrained=False):
    missing = (
        _model_prerequisites(config, purpose, allow_untrained)
        + _native_prerequisites(config, purpose)
        + _data_prerequisites(config)
    )
    if missing:
        raise MissingPrerequisites(sorted(set(missing)))


def metadata(config=None, status="unknown"):
    def git(*args):
        return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True).strip()

    import shutil

    gpu = "unavailable"
    if shutil.which("nvidia-smi"):
        gpu = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.used,utilization.gpu", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    source = hashlib.sha256()
    for file in sorted((ROOT / "cp_adaptok").rglob("*.py")):
        source.update(str(file.relative_to(ROOT)).encode())
        source.update(file.read_bytes())
    return {
        "time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "timezone": "Asia/Shanghai",
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_sha256": source.hexdigest(),
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "gpu_snapshot": gpu,
        "gpu_shared": "not assumed isolated; see gpu_snapshot and serial benchmark commands",
        "status": status,
        "config": config,
        "upstreams": json.loads((ROOT / "third_party/manifest.json").read_text()),
    }


def write_json(path, data):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def report_missing(path, error, config):
    report = metadata(config, "blocked_prerequisites")
    report.update(conditions=error.conditions, executed_training=False, trained_rd_validated=False)
    write_json(path, report)
    print(
        json.dumps(
            {"status": report["status"], "conditions": error.conditions, "report": str(resolve(path))},
            ensure_ascii=False,
        )
    )


def save_checkpoint(path, model, optimizer, generator, step, config, status):
    path = resolve(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "format_version": 1,
            "model_config": asdict(model.config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config,
            "status": status,
            "sampling_rng": generator.get_state(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            "python_rng": random.getstate(),
            "vq_training_state": model.vq.training_state(),
            "metadata": metadata(config, status),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(path):
    checkpoint = torch.load(resolve(path), map_location="cpu", weights_only=True)
    if checkpoint.get("format_version") != 1:
        raise ValueError("unsupported checkpoint format")
    return checkpoint


def _load_requested_artifact(config, checkpoint_override):
    paths = config.get("paths", {})
    checkpoint_path = checkpoint_override or paths.get("checkpoint")
    if checkpoint_override and not resolve(checkpoint_override).is_file():
        raise MissingPrerequisites([f"explicit checkpoint={checkpoint_override}"])
    artifact = (
        load_checkpoint(checkpoint_path) if checkpoint_path and resolve(checkpoint_path).is_file() else None
    )
    return artifact, checkpoint_path


def _configure_model(config, artifact):
    model_config = CodecConfig(**(artifact["model_config"] if artifact else config.get("model", {})))
    if artifact and asdict(model_config) != asdict(CodecConfig(**config.get("model", {}))):
        raise ValueError("checkpoint architecture differs from the requested config")

    def existing(name):
        path = config.get("paths", {}).get(name)
        return resolve(path) if path and resolve(path).is_file() else None

    model = CausalProgressiveCodec(model_config, existing("ld_checkpoint"), existing("intra_checkpoint"))
    control = config.get("control", "learned_query")
    if control == "fixed_spatial":
        from cp_adaptok.baselines.fixed_spatial import FixedSpatialTokenizer

        model.tokenizer = FixedSpatialTokenizer(model_config.n_max, model_config.dim)
    elif control != "learned_query":
        raise ValueError("unknown experimental control")
    return model, existing, control


def _initialization_status(model, artifact, existing):
    if artifact:
        model.load_state_dict(artifact["model"], strict=True)
        settings = artifact["config"].get("training", {}).get("vq", {})
        if settings.get("strategy") == "continuous" or artifact["step"] < settings.get("warmup_steps", 0):
            model.vq.set_training_policy("continuous")
        return artifact["status"]
    return (
        "initialized_pretrained_backbones"
        if existing("ld_checkpoint") and existing("intra_checkpoint")
        else "random_weights_structure_only"
    )


def build_model(config, device, allow_untrained=False, checkpoint_override=None):
    artifact, checkpoint_path = _load_requested_artifact(config, checkpoint_override)
    model, existing, control = _configure_model(config, artifact)
    status = _initialization_status(model, artifact, existing)
    requires_trained = artifact is not None or config.get("purpose") != "train"
    if not allow_untrained and status != "trained" and requires_trained:
        raise MissingPrerequisites(
            [f"CP model status={status}; use --allow-untrained explicitly for structural measurements"]
        )
    provenance = {
        "checkpoint": str(resolve(checkpoint_path)) if artifact else None,
        "checkpoint_step": artifact["step"] if artifact else None,
        "initialization_status": status,
        "effective_model_config": asdict(model.config),
        "control": control,
    }
    model.artifact_provenance = provenance
    return model.to(device), status, artifact
