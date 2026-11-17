"""Absolute quality comparison is permitted only after exact frame, pixel and metric-weight checks."""

import json
from pathlib import Path

from .artifacts import write_csv, write_json
from .metrics import aggregate


def load(path):
    return json.loads(Path(path).read_text())


def frame_key(row):
    return row["content_id"], row["frame_id"]


def official_rows(directory, budget):
    videos = [load(p) for p in sorted(Path(directory).glob("*/completed.json"))]
    rows = [r for v in videos for w in v["windows"] for r in w["budgets"][str(budget)]["frames"]]
    return videos, rows


def check_pixel_parity(root, videos):
    parity = load(root / "cp_input_parity.json")
    expected = {(w["content_id"], w["start"]): w["rgb_sha256"] for w in parity["windows"]}
    actual = {(v["source"]["content_id"], w["input"]["start"]): w["input"]["rgb_sha256"] for v in videos for w in v["windows"]}
    if expected != actual:
        raise ValueError("official evaluation and CP native interpreter received different RGB pixels/windows")
    weights = load(root / "metric_manifest.json")["weight_sha256"]
    official = {Path(p).name: sha for p, sha in weights.items()}
    for p, sha in parity["metric_weight_sha256"].items():
        if official.get(Path(p).name) != sha:
            raise ValueError(f"CP perceptual metric weights differ: {p}")
    return {"windows": len(actual), "source_frames": parity["frame_count"], "rgb_sha256_equal": True,
            "metric_weight_sha256_equal": True, "cp_metric_versions": parity["packages"]}


def compare(root, reference):
    root, reference = Path(root), Path(reference)
    videos, rows = official_rows(root / "final_1080p", 2048)
    parity = check_pixel_parity(root, videos)
    keys = {frame_key(r) for r in rows}
    if len(keys) != len(rows):
        raise ValueError("duplicate official frame measurements")
    comparisons = historical_rows(reference, keys) + ablation_rows(root, keys)
    write_csv(root / "absolute_quality_comparison.csv", comparisons)
    result = {"comparison": "absolute reconstruction quality only; not equal-rate RD", "parity": parity,
              "quality_only": True, "bpp": None, "rows": comparisons}
    write_json(root / "absolute_quality_comparison.json", result)
    return result


def historical_rows(reference, keys):
    comparisons = []
    for branch in ["B0", "B1"]:
        report = load(reference / branch / "validation.json")
        historical_keys = {(r["source"]["content_id"], r["frame_id"]) for r in report["records"]}
        if historical_keys != keys:
            raise ValueError("absolute quality comparison requires exactly the same held-out frames")
        for budget in [128, 512, 1024, 2048]:
            selected = [{"content_id": r["source"]["content_id"], **next(p for p in r["prefixes"] if p["tokens"] == budget)} for r in report["records"]]
            comparisons.extend(comparison_rows(branch, f"{budget}/frame", selected))
    return comparisons


def ablation_rows(root, keys):
    comparisons = []
    for name, directory in [("Official stride128", "final_1080p"), ("Official stride96", "token_ablation/overlap")]:
        for budget in [512, 1024, 2048]:
            _, selected = official_rows(root / directory, budget)
            if {frame_key(r) for r in selected} != keys:
                raise ValueError("official ablation used different frames")
            comparisons.extend(comparison_rows(name, f"{budget}/tile/native-window", selected))
    return comparisons


def comparison_rows(model, units, frames):
    grouped = [("all", frames)] + [(identity, [r for r in frames if r["content_id"] == identity]) for identity in sorted({r["content_id"] for r in frames})]
    return [{"model": model, "budget_units": units, "content_id": identity, **aggregate(rows)} for identity, rows in grouped]
