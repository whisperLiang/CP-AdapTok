"""Small shared boundary utilities for reproducible experiment artifacts."""

import hashlib
import json
import platform
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch
import yaml


def stable_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2**20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_config(path):
    return yaml.safe_load(Path(path).read_text())


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def environment():
    return {"time": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(), "timezone": "Asia/Shanghai",
            "git": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
            "source_sha256": source_identity(),
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
            "python_trace_enabled": sys.gettrace() is not None,
            "branch": subprocess.check_output(["git", "branch", "--show-current"], text=True).strip(),
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
            "driver": subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                               text=True).strip() if torch.cuda.is_available() else None}


def source_identity():
    root = Path(__file__).parent
    return stable_hash({str(p.relative_to(root)): file_hash(p) for p in sorted(root.rglob("*.py"))
                        if "tests" not in p.parts})
