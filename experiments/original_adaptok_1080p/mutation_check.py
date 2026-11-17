"""Targeted behavioral mutations in disposable copies; never mutate the experiment in use."""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .artifacts import write_json

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = Path("experiments/original_adaptok_1080p")
MUTATIONS = [
    ("tail_first_frame", "video_reader.py", "real[-1:].expand", "real[:1].expand", "test_temporal.py"),
    ("replicate_padding", "tiling.py", 'mode="reflect"', 'mode="replicate"', "test_tiling.py"),
    ("discard_group_divisibility", "model_adapter.py", "or budget % groups", "or False", "test_reconstruction.py"),
    ("conflate_psnr_aggregation", "metrics.py", 'psnr_from_global_mse_db=psnr(result["mse"])',
     'psnr_from_global_mse_db=sum(r["psnr"] for r in rows) / len(rows)', "test_metrics.py"),
    ("drop_extra_csv_metrics", "artifacts.py", "list(dict.fromkeys(key for row in rows for key in row))", "list(rows[0])", "test_artifacts.py"),
    ("encode_twice", "reconstruction.py", "        decode_batch(adapter, encoded, coordinates, writers, timing)",
     "        decode_batch(adapter, adapter.encode(tiles), coordinates, writers, timing)", "test_reconstruction.py"),
]


def check_one(spec, output):
    name, filename, before, after, test = spec
    with tempfile.TemporaryDirectory(prefix="official_adaptok_mutation_") as temporary:
        checkout = Path(temporary)
        shutil.copytree(ROOT / "experiments", checkout / "experiments", ignore=shutil.ignore_patterns("__pycache__", ".pytest_cache"))
        target = checkout / PACKAGE / filename
        original = target.read_text()
        if before not in original:
            raise ValueError(f"missing mutation target {name}")
        target.write_text(original.replace(before, after, 1))
        env = {**os.environ, "PYTHONPATH": str(checkout) + os.pathsep + str(ROOT), "OMP_NUM_THREADS": "4"}
        process = subprocess.run([sys.executable, "-m", "pytest", str(PACKAGE / "tests" / test), "-q", "--tb=short"], cwd=checkout, env=env, capture_output=True, text=True)
        text = process.stdout + process.stderr
        (output / f"{name}.txt").write_text(text)
        killed = process.returncode == 1 and " failed" in text and "ERROR collecting" not in text
        return {"mutation": name, "killed": killed, "returncode": process.returncode}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "results/original_adaptok_1080p/mutations")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = [check_one(spec, args.output) for spec in MUTATIONS]
    write_json(args.output / "summary.json", results)
    print(json.dumps(results, indent=2))
    if not all(r["killed"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
