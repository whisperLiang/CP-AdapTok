"""Capture local assets and verify isolation without opening sealed test pixels."""

import importlib.metadata
import json
import subprocess
from datetime import datetime
from zoneinfo import ZoneInfo

import torch

from aip_vtok.runtime import ROOT, file_hash, resolve, write_json
from aip_vtok.training.data import sources

PROTECTED = (
    "cp_adaptok",
    "adaptok_hr",
    "third_party",
    "experiments/fixed2048",
    "results/fixed2048",
    "configs/fixed2048.yaml",
)


def command(*args):
    result = subprocess.run(args, cwd=ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def audit(config):
    train, hashes = sources(config, "train")
    validation, _ = sources(config, "validation")
    packages = {}
    for name in ("torch", "numpy", "pytest", "lpips", "DISTS-pytorch", "pytorch-msssim"):
        packages[name] = importlib.metadata.version(name)
    test = json.loads(resolve(config["manifest_dir"] + "/cpat_test.json").read_text())["records"]
    changed = command("git", "diff", "630e814", "--name-only", "--", *PROTECTED)
    if changed:
        raise RuntimeError(f"protected tracked files changed: {changed}")
    artifact = {
        "date_local": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "branch": command("git", "branch", "--show-current"),
        "base_commit": command("git", "rev-parse", "630e814"),
        "initial_working_branch": "experiment/adaptok-hr-perceptual-recovery",
        "git_status": command("git", "status", "--short"),
        "official_source": command("git", "submodule", "status"),
        "gpu": command(
            "nvidia-smi", "--query-gpu=name,memory.total,memory.used,utilization.gpu", "--format=csv"
        ),
        "gpu_processes": command(
            "nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"
        ),
        "packages": packages,
        "cuda": torch.version.cuda,
        "manifest_hashes": hashes,
        "native_exact_train": {"videos": len(train), "frames": sum(r["original_frames"] for r in train)},
        "native_exact_validation": {
            "videos": len(validation),
            "frames": sum(r["original_frames"] for r in validation),
        },
        "sealed_test_metadata_only": [r["content_id"] for r in test if r["resolution"] == [1920, 1080]],
        "weights_sha256": {k: file_hash(config[k]) for k in ("intra_checkpoint", "ld_checkpoint")},
        "protected_diff": changed,
        "config": config,
        "scope": "Gate 0; no test pixels, no network transport, no ABR",
    }
    write_json(resolve(config["output"]) / "audit.json", artifact)
    return artifact


if __name__ == "__main__":
    from aip_vtok.runtime import load_config

    print(json.dumps(audit(load_config("configs/aip_vtok/gate0.yaml")), ensure_ascii=False, indent=2))
