"""Freeze the experiment's upstream, older endpoints, and local native build."""

import argparse
import json
import platform
import subprocess
from pathlib import Path
from typing import Any

from hla_tok.evaluation.data import load_manifests
from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import metric_hashes, sha256_file
from scripts.hla_tok.v2.protocol import ROOT

CONFIG = ROOT / "configs/hla_tok/hts"
UPSTREAM = ROOT / "third_party/DCVC"
V2_CHECKPOINT = "checkpoints/hla_tok/uf_v2/d1-joint-seed2-step20000.pt"
V3_CHECKPOINT = "checkpoints/hla_tok/uf_v3/r0-probe-seed1-step2001.pt"
V3_CDF = "results/hla_tok/uf_v3/entropy/static_cdf.json"
V3_ASSETS = (V3_CHECKPOINT, V3_CDF, "results/hla_tok/uf_v3/evaluation/A2/summary.json",
             "results/hla_tok/uf_v3/evaluation/A4/summary.json", "results/hla_tok/uf_v3/evaluation/rates.json",
             "results/hla_tok/uf_v3/final_report.md", "results/hla_tok/compact/original_rd.csv",
             "results/hla_tok/compact/original_window_ledger.csv", "results/hla_tok/temporal/sequence_manifest.json",
             "configs/hla_tok/validation_cohort.json", "results/original_adaptok_1080p/checkpoint_manifest.json")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def capture(*cmd: str) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def clean_checkout(root: Path, expected: str) -> None:
    if capture("git", "-C", str(root), "rev-parse", "HEAD") != expected:
        raise ValueError(f"checkout moved: {root}")
    if capture("git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"):
        raise ValueError(f"tracked assets changed: {root}")


def verified_weights(root: Path) -> dict[str, dict]:
    identity = read_json(CONFIG / "official_weights.json")["weights"]
    for spec in identity.values():
        path = root / spec["filename"]
        if path.stat().st_size != spec["bytes"] or sha256_file(path) != spec["sha256"]:
            raise ValueError(f"weight differs from the published lock: {path}")
    return identity


def old_identity(args) -> dict:
    clean_checkout(args.v2, "ffbb7177e869b49934bcddf37c24fc7eade8bf41")
    clean_checkout(args.v3, "5a415c98a57c4198f2e792974c14275217ad5952")
    paths = [args.v2 / V2_CHECKPOINT, *(args.v3 / item for item in V3_ASSETS)]
    original = read_json(args.v3 / "results/original_adaptok_1080p/checkpoint_manifest.json")
    paths.extend(Path(original["directory"]) / name for name in ("model.safetensors", "config.json"))
    paths.extend(args.previous_weights / name for name in ("cvpr2026_image.pth.tar", "cvpr2026_video_ld.pth.tar"))
    for directory in ("native", "pack-reuse", "records", "legacy"):
        paths.extend(sorted((args.v3 / ".build/uf_v3" / directory).glob("*.so")))
    hashes = {str(path): sha256_file(path) for path in paths}
    if hashes[str(args.v2 / V2_CHECKPOINT)] != read_json(
            args.v3 / "results/hla_tok/uf_v3/evaluation/A2/summary.json")["checkpoint_sha256"]:
        raise ValueError("the V2 Seed 2 archive does not match its checkpoint")
    if hashes[str(args.v3 / V3_CHECKPOINT)] != read_json(
            args.v3 / "results/hla_tok/uf_v3/evaluation/A4/summary.json")["checkpoint_sha256"]:
        raise ValueError("the V3 A4 archive does not match its checkpoint")
    return hashes


def main() -> None:
    import inference_extensions_cuda as inference
    import MLCodec_extensions_cpp as rans
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--previous-weights", type=Path, required=True)
    parser.add_argument("--cutlass", type=Path, required=True)
    parser.add_argument("--nvcc", type=Path, required=True)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--v3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verify-previous", action="store_true")
    args = parser.parse_args()
    pinned = read_json(CONFIG / "upstream.json")
    clean_checkout(UPSTREAM, pinned["commit"])
    clean_checkout(args.cutlass, pinned["cutlass"]["commit"])
    gpu = torch.cuda.get_device_name()
    if torch.cuda.device_count() != 1 or gpu != pinned["build"]["gpu"]:
        raise ValueError("select the pinned single H100 target device")
    _, manifests = load_manifests(ROOT / "data/manifests")
    metrics = metric_hashes()
    if metrics != read_json(ROOT / "configs/hla_tok/validation_cohort.json")["metric_weights"]:
        raise ValueError("metric hash differs from previous experiments")
    identity = {
        "official_commit": pinned["commit"], "cutlass_commit": pinned["cutlass"]["commit"],
        "previous_assets_sha256": old_identity(args), "metric_weights_sha256": metrics,
        "frozen_dataset_manifest_sha256": manifests, "weights": verified_weights(args.weights),
        "extensions_sha256": {module.__name__: sha256_file(Path(module.__file__)) for module in (rans, inference)},
        "extension_sources_patch_sha256": sha256_file(CONFIG / "native_state_audit.patch"),
        "python": platform.python_version(), "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "nvcc": capture(str(args.nvcc), "--version"), "compiler": capture("c++", "--version").splitlines()[0],
        "gpu": gpu, "compute_capability": list(torch.cuda.get_device_capability()), "sm": "90a",
        "nvidia_smi": capture("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv"),
        "protocol_sha256": sha256_file(CONFIG / "protocol.json"),
    }
    if args.verify_previous:
        previous = read_json(args.output)
        if previous != identity:
            raise ValueError("experiment asset identity drifted")
    else:
        write_json(args.output, identity)


if __name__ == "__main__":
    main()
