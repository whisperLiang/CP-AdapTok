"""Explicit checkpoint freeze and one-shot, native-only sealed Test evaluation."""

import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from hla_tok import BUDGETS
from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.data import load_manifests
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import FullFrameQuality, sha256_file
from hla_tok.evaluation.reconstruction import CodeUsage
from scripts.hla_tok.v2.acceptance import record_test_counts
from scripts.hla_tok.v2.protocol import ROOT, frozen_data, json_at, setup
from scripts.hla_tok.v3.downstream import _approved_checkpoint, model_from_checkpoint
from scripts.hla_tok.v3.experiments import on_gpu, require_pass, state_of
from scripts.hla_tok.v3.protocol import NATIVE, OUTPUT, PROTOCOL
from scripts.hla_tok.v3.quality import _bootstrap, _native_reconstructions, _summaries

FREEZE = OUTPUT / "final_freeze.json"


def _test_scope():
    _, _, identity = frozen_data()
    manifests, hashes = load_manifests(ROOT / "data/manifests")
    if hashes != identity["manifest_hashes"]:
        raise ValueError("sealed Test manifest differs from the frozen provenance")
    rows = manifests["test"]
    if any(row["role"] != "test" for row in rows):
        raise ValueError("a sealed record has a non-Test role")
    native = [row for row in rows if row["resolution"] == [NATIVE.width, NATIVE.height] and not row["resized"]]
    if not native:
        raise ValueError("the sealed manifest has no unresized native inputs")
    columns = {row["content_id"]: len(row["frame_hashes"]) for row in native}
    return native, {"test_manifest_sha256": hashes["test"], "native_videos": columns,
                    "frames": sum(columns.values()), "non_native_videos": len(rows) - len(native),
                    "policy": "every frame of every unresized original 1920x1080 Test sequence; no rescaling"}


def _prerequisites(config):
    precision = "bf16"
    for kind in ("cpu", "cuda"):
        audit = record_test_counts(OUTPUT / f"{kind}_tests.xml")
        if not audit or audit["failed"] or not audit["passed"]:
            raise RuntimeError(f"the {kind} regression has not passed")
    require_pass(OUTPUT / "mutations" / "summary.json")
    models = []
    for seed in (1, 2):
        require_pass(state_of("quantized", seed))
        progressive = require_pass(state_of("progressive", seed))
        path = _approved_checkpoint(progressive)
        performance = progressive["performance"]
        if performance["precision"] != precision or performance["pipeline_fps"] < config["performance"]["pipeline_fps"]:
            raise RuntimeError("final BF16 offline pipeline did not clear the registered gate")
        tests = OUTPUT / "progressive" / f"seed{seed}" / "D" / "verify" / "native_verify.json"
        require_pass(tests)
        _, state = model_from_checkpoint(path)
        models.append({"seed": seed, "path": str(path), "sha256": sha256_file(path),
                       "architecture": state["counters"]["architecture"], "step": state["step"],
                       "training_identity": state["identity"]})
    return models


def freeze_final(config):
    models = _prerequisites(config)
    _, scope = _test_scope()
    record = {"status": "PASS", "models": models, "test_scope": scope, "protocol_sha256": sha256_file(PROTOCOL),
              "training_loss_sha256": sha256_file(OUTPUT / "gradient_calibration" / "dense.json")}
    if FREEZE.exists():
        if json_at(FREEZE) != record:
            raise RuntimeError("an existing model freeze is immutable")
    else:
        write_json(FREEZE, record)
    return record


def _read_native(row, item):
    root = Path(row["path"]).resolve()
    candidate = (root / item["file"]).resolve()
    if not candidate.is_relative_to(root):
        raise PermissionError("sealed frame escapes its frozen source path")
    data = candidate.read_bytes()
    if hashlib.sha256(data).hexdigest() != item["sha256"]:
        raise ValueError("sealed source hash differs")
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None or decoded.shape[:2] != (NATIVE.height, NATIVE.width):
        raise ValueError("sealed Test input is not native 1080p")
    return torch.from_numpy(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)).permute(2, 0, 1)


@torch.inference_mode()
def _score_test(model, sources, scope, quality, output):
    statistics, code_usage = defaultdict(list), {budget: CodeUsage(8192) for budget in BUDGETS}
    attempted = 0
    for row in sources:
        for index, item in enumerate(row["frame_hashes"]):
            attempted += 1
            write_json(output / "pixel_reads.json", {"test": attempted, "train": 0, "validation": 0})
            source = transfer(_read_native(row, item)[None])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                images = list(_native_reconstructions(model, source, BUDGETS, True, False))
            for (mode, budget), image, rate, _, ids in images:
                code_usage[budget].update(ids)
                statistics[(mode, budget)].append(
                    {"video": row["content_id"], "frame": index, "mode": mode, "budget": budget}
                    | rate | quality(source, image))
    if any(len(frames) != scope["frames"] for frames in statistics.values()):
        raise RuntimeError("the one-shot sealed evaluation did not consume the complete native scope")
    for (mode, budget), rows in statistics.items():
        _bootstrap(rows, output / f"video_uncertainty_{mode}_{budget}.json")
    summary = [{"split": "test"} | row for row in _summaries(statistics, code_usage, output, True)]
    write_csv(output / "per_frame.csv", [row for rows in statistics.values() for row in rows])
    write_json(output / "metrics.json", summary)
    return summary


def evaluate_sealed(config):
    if not FREEZE.is_file():
        raise PermissionError("Test remains sealed until an explicit final checkpoint freeze")
    frozen = require_pass(FREEZE)
    if freeze_final(config) != frozen:
        raise PermissionError("freeze does not match the final passing checkpoints")
    sources, scope = _test_scope()
    _, _, identity = frozen_data()
    results = {}
    for entry in frozen["models"]:
        setup(entry["seed"])
        model, _ = model_from_checkpoint(Path(entry["path"]))
        output = OUTPUT / "sealed_test" / entry["sha256"]
        if (output / "result.json").is_file():
            results[entry["seed"]] = json_at(output / "result.json")
            continue
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / "authorization.json", {"frozen_model": entry, "scope": scope})
        write_json(output / "pixel_reads.json", {"test": 0, "train": 0, "validation": 0})
        quality = FullFrameQuality("cuda", identity["metric_weights"])
        rows = _score_test(on_gpu(model).eval(), sources, scope, quality, output)
        result = {"status": "PASS", "evaluation_completed": True, "seed": entry["seed"],
                  "checkpoint_sha256": entry["sha256"], "frames": scope["frames"], "metrics": rows}
        write_json(output / "result.json", result)
        results[entry["seed"]] = result
    return results
