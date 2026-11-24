"""Read-only asset verification and the original R2 evaluator."""

import subprocess
from importlib.metadata import version
from pathlib import Path

import torch

from hla_tok import HLATok
from hla_tok.evaluation.io import environment, write_json
from hla_tok.evaluation.metrics import FullFrameQuality, metric_hashes, sha256_file
from hla_tok.training.v2 import load_experimental_checkpoint
from scripts.hla_tok.v2.acceptance import git_snapshot
from scripts.hla_tok.v2.protocol import COHORT, ROOT, frozen_data, json_at, setup
from scripts.hla_tok.v2.train import native_score
from scripts.hla_tok.v3.protocol import BASE, PROTOCOL

V2_RESULTS = Path("results/hla_tok/v2")


def r2_source(v2: Path) -> tuple[Path, dict]:
    acceptance = json_at(v2 / V2_RESULTS / "acceptance.json")
    if acceptance["D2"]["status"] != "FAIL" or acceptance["phase2_eligible"]:
        raise ValueError("V2 D2 freeze record was changed")
    reference, = (row for row in acceptance["D2"]["paired_comparisons"] if row["arm"] == "R2")
    manifest, = (row for row in json_at(v2 / V2_RESULTS / "checkpoint_manifest.json")
                 if row["path"] == reference["checkpoint"])
    path = v2 / reference["checkpoint"]
    if not path.is_file() or sha256_file(path) != manifest["sha256"] or manifest["step"] != 2000:
        raise ValueError(f"missing or mismatched V2 R2 step-2000 checkpoint: {path}")
    return path, reference


def verify_assets(v2: Path):
    legacy = json_at(v2 / V2_RESULTS / "legacy_assets_final.json")
    assets = dict(legacy["asset_sha256"])
    for row in json_at(v2 / V2_RESULTS / "checkpoint_manifest.json"):
        assets[str(v2 / row["path"])] = row["sha256"]
    command = ["git", "-C", str(v2)]
    if subprocess.check_output([*command, "rev-parse", "HEAD"], text=True).strip() != BASE:
        raise ValueError("V2 worktree does not point to the specified base commit")
    paths = subprocess.check_output(
        [*command, "ls-tree", "-r", "--name-only", BASE, str(V2_RESULTS), "docs/hla_tok/v2", "configs/hla_tok"],
        text=True,
    ).splitlines()
    from hashlib import sha256

    for relative in paths:
        expected = sha256(subprocess.check_output([*command, "show", f"{BASE}:{relative}"])).hexdigest()
        for root in (v2, ROOT):
            assets[str(root / relative)] = expected
    for name, digest in assets.items():
        if sha256_file(Path(name)) != digest:
            raise ValueError(f"frozen asset differs: {name}")
    return assets


def preflight(v2: Path, destination: Path):
    setup()
    reader, cohort, identity = frozen_data()
    if metric_hashes() != identity["metric_weights"]:
        raise ValueError("perceptual weight hashes differ from the frozen evaluation")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("the registered V3 protocol requires CUDA BF16")
    hardware = environment()
    if torch.cuda.device_count() != 1 or "H100 PCIe" not in hardware["gpu"]:
        raise RuntimeError(f"expected one H100 PCIe; found {hardware['gpu']}")
    source, _ = r2_source(v2)
    assets = verify_assets(v2)
    result = {
        "status": "PASS", "base": BASE, "v2_d2": "FAIL", "phase2_eligible_v2": False,
        "v2": git_snapshot(v2), "v3": git_snapshot(ROOT),
        "hardware": hardware | {"total_memory_bytes": torch.cuda.get_device_properties(0).total_memory},
        "dependencies": {name: version(name) for name in ("torch", "torchvision", "numpy", "PyYAML", "Pillow",
                                                          "lpips", "DISTS-pytorch", "pytorch-msssim")},
        "protocol_sha256": sha256_file(PROTOCOL), "cohort_sha256": sha256_file(COHORT), "cohort": cohort,
        "manifest_sha256": reader.hashes, "metric_weight_sha256": identity["metric_weights"],
        "fixed13_sha256": sha256_file(ROOT / "hla_tok/streaming/fixed13.py"),
        "r2_initializer": {"path": str(source), "sha256": assets[str(source)]},
        "asset_sha256": assets, "pixel_reads": reader.counts,
    }
    write_json(destination, result)
    return result


def baseline_replay(v2: Path, output: Path):
    setup()
    output.mkdir(parents=True, exist_ok=False)
    reader, cohort, identity = frozen_data()
    source, ref = r2_source(v2)
    model = HLATok("R2")
    load_experimental_checkpoint(source, model, sha256_file(COHORT))
    model.cuda().to(memory_format=torch.channels_last)
    quality = FullFrameQuality("cuda", identity["metric_weights"])
    try:
        row = native_score(model, reader, cohort, quality, output, quantized=False, subset=False)
    finally:
        write_json(output / "pixel_reads.json", reader.counts)
    observed = {key: row[key] for key in ("global_psnr", "lpips", "dists")}
    exact = row["frames"] == 174 and all(observed[key] == ref[key] for key in observed)
    result = {"status": "PASS" if exact else "FAIL", "frames": row["frames"],
              "source_sha256": sha256_file(source),
              "recorded": {key: ref[key] for key in observed}, "replayed": observed,
              "pixel_reads": reader.counts, "path": str(output)}
    write_json(output / "replay.json", result)
    if not exact:
        raise RuntimeError("strict V2 R2 replay differs from the frozen native result")
    return result
