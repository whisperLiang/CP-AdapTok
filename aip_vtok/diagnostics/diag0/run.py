"""Reproducible single-card entry point with read-only access to sealed inputs."""

import argparse
import importlib.metadata
import os
import random
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import torch

from aip_vtok.evaluation.fix import load_evaluation_codec
from aip_vtok.runtime import ROOT, file_hash, resolve, write_json
from aip_vtok.training.determinism import configure_determinism
from aip_vtok.training.fix import CONFIG, fix_settings
from cp_adaptok.metrics import MetricSuite

from .fit import (
    CALIBRATION_STRIDE,
    FIT_SEED,
    RANKS,
    SENSITIVITY_WINDOWS,
    TRAIN_WINDOW,
    fit_probes,
    load_fits,
    save_fits,
)
from .io import (
    BASE,
    FIX,
    FIX_WEIGHTS,
    GATE,
    OUTPUT,
    Catalog,
    command,
    gpu_snapshot,
    json_at,
    metric_weight_hashes,
    native_control,
    protected_artifacts,
    protected_code_changes,
    run_lock,
    sealed_projector,
)
from .matched import run_matched
from .packets import LAYOUTS
from .sparse import BUDGETS, DEPTHS, HEADER_BYTES, LEVELS, QUANTILE, QUANTILE_TANH_INPUT, REF_BUDGET

SEED = 7301
STEP = 1000
FIT_PATH = OUTPUT / "train_fit.pt"
PROVENANCE = OUTPUT / "provenance.json"


def input_identity(config, manifest_hashes):
    fix = FIX_WEIGHTS.relative_to(resolve("checkpoints/aip_vtok/gate1_fix"))
    expected = json_at(GATE / "gate1_fix/checkpoints_manifest.json")[str(fix)]["sha256"]
    actual = file_hash(FIX_WEIGHTS)
    if actual != expected:
        raise RuntimeError("Fix-A step 1000 differs from the sealed checkpoint")
    return {
        "base_commit": BASE,
        "train_manifest": manifest_hashes["train"],
        "alignment_sha256": file_hash(resolve(config["checkpoints"]) / "alignment.pt"),
        "intra_sha256": file_hash(config["intra_checkpoint"]),
        "ld_sha256": file_hash(config["ld_checkpoint"]),
        "projector_sha256": sealed_projector()["sha256"],
        "fix_a_sha256": actual,
        "fit_seed": FIT_SEED,
        "fit_window": TRAIN_WINDOW,
        "sensitivity_windows": SENSITIVITY_WINDOWS,
        "calibration_stride": CALIBRATION_STRIDE,
        "calibration_quantile": QUANTILE,
        "calibration_tanh_input": QUANTILE_TANH_INPUT,
        "qp": config["qp"],
        "gop": config["gop"],
    }


def configure():
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    configure_determinism()
    config = fix_settings("fix_a", 1)
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    return config


def protocol(config, identity, train, plan):
    declarations = (CONFIG, "configs/aip_vtok/gate1.yaml", "configs/aip_vtok/gate0.yaml")
    sources = (GATE / "native_bitstream_control.json", FIX / "diagnostics.json", FIX / "tokenizer_oracle.json")
    return {
        "identity": identity,
        "effective_frozen_config": config,
        "configuration_sha256": {str(path): file_hash(path) for path in declarations},
        "metric_weights_sha256": metric_weight_hashes(),
        "historical_evidence_sha256": {str(p.relative_to(ROOT)): file_hash(p) for p in sources},
        "manifests_metadata_hashes": train.manifest_hashes,
        "train_windows": plan,
        "test_images_read": 0,
        "training_steps": 0,
        "rng_seed": SEED,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cuda_workspace": os.environ["CUBLAS_WORKSPACE_CONFIG"],
        "tf32": False,
        "dmc_ld_actual_bitstream": native_control(),
        "packages": {
            package: importlib.metadata.version(package)
            for package in ("torch", "torchvision", "numpy", "lpips", "DISTS-pytorch", "matplotlib")
        },
        "cuda_version": torch.version.cuda,
        "official_dcvc_gitlink": command("git", "rev-parse", f"{BASE}:third_party/DCVC"),
        "official_dcvc_checked_out": command("git", "-C", "third_party/DCVC", "rev-parse", "HEAD"),
        "probes": {
            "ranks": RANKS, "budgets": BUDGETS, "fsq_levels": LEVELS, "depths": DEPTHS.tolist(),
            "layouts": LAYOUTS, "reference_budget": {"I": REF_BUDGET[True], "P": REF_BUDGET[False]},
            "headers_est_bytes": HEADER_BYTES, "bilinear_align_corners": False, "bilinear_antialias": False,
        },
    }


def finish_stage(stage, before, projector, metadata):
    after = protected_artifacts()
    code_changes = protected_code_changes()
    check = {
        "stage": stage,
        "protected_artifacts_before": before,
        "protected_artifacts_after": after,
        "protected_code_changes": code_changes,
        "projector_before": projector,
        "projector_after": sealed_projector(),
        "gpu_end": gpu_snapshot(),
        "finished_asia_shanghai": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
    }
    if before != after or code_changes or projector != check["projector_after"]:
        raise RuntimeError(f"protected artifacts changed: {check}")
    metadata.setdefault("integrity", {})[stage] = check
    write_json(PROVENANCE, metadata)


def obtain_fits(stage, train, backbone, identity):
    if stage != "fit" and FIT_PATH.is_file():
        return load_fits(FIT_PATH, identity)
    if stage == "closed":
        raise RuntimeError("complete the matched stage before closed-loop evaluation")
    fits, plan = fit_probes(train, backbone)
    save_fits(FIT_PATH, fits, plan, identity)
    print(f"saved Train-only fits {file_hash(FIT_PATH)}", flush=True)
    return fits, plan


def render_report():
    from .report import generate_report

    generate_report()


def open_provenance(stage):
    metadata = json_at(PROVENANCE) if PROVENANCE.is_file() else {}
    checks = metadata.setdefault("integrity", {})
    if stage == "closed" and "matched" not in checks:
        raise ValueError("closed loop requires a completed, audited D0-D artifact")
    invalidated = {
        "fit": ("fit", "matched", "closed"), "all": ("matched", "closed"),
        "matched": ("matched", "closed"), "closed": ("closed",),
    }
    for name in invalidated[stage]:
        checks.pop(name, None)
    write_json(PROVENANCE, metadata)
    return metadata


def run(stage):
    start = gpu_snapshot()
    config = configure()
    before, projector = protected_artifacts(), sealed_projector()
    train = Catalog.open(config, "train")
    identity = input_identity(config, train.manifest_hashes)
    codec = load_evaluation_codec(config, FIX, FIX_WEIGHTS, STEP).requires_grad_(False)
    metadata = open_provenance(stage)
    fits, plan = obtain_fits(stage, train, codec.backbone, identity)
    metadata.update(protocol(config, identity, train, plan))
    metadata.update(train_fit_sha256=file_hash(FIT_PATH), gpu_start=start)
    if stage == "fit":
        finish_stage(stage, before, projector, metadata)
        return

    suite = MetricSuite("cuda:0", names=("psnr", "lpips", "dists"))
    if suite.errors:
        raise RuntimeError(f"mandatory full-resolution metrics unavailable: {suite.errors}")
    validation = Catalog.open(config, "validation")
    if stage in ("all", "matched"):
        videos = config["stage1"]["validation"]["videos"]
        metadata["validation_frame_ids"] = run_matched(validation, codec, suite, fits, videos)
        finish_stage("matched", before, projector, metadata)
    if stage in ("all", "closed"):
        from .closed import run_closed

        run_closed(validation, codec, suite, fits)
        finish_stage("closed", before, projector, metadata)
    if stage == "all":
        render_report()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "fit", "matched", "closed", "report"), default="all")
    args = parser.parse_args()
    with run_lock():
        if args.stage == "report":
            render_report()
        else:
            run(args.stage)
