"""Hash read-only provenance and checkpoint assets without opening sealed pixels."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from hla_tok.evaluation.data import AuditedFrames, load_cohort
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file


def repository(path):
    head = subprocess.check_output(["git", "-C", str(path), "rev-parse", "HEAD"], text=True).strip()
    diff = subprocess.check_output(["git", "-C", str(path), "diff", "--binary", "HEAD", "--"])
    status = subprocess.check_output(["git", "-C", str(path), "status", "--porcelain"], text=True).splitlines()
    return {"commit": head, "tracked_diff_sha256": hashlib.sha256(diff).hexdigest(), "status": status}


def collect_assets(original, joint):
    selection_path = joint / "results/aip_vtok/joint_v1/seed1/selection.json"
    selection = json.loads(selection_path.read_bytes())
    frozen = selection["identity"]["frozen_file_sha256"]
    assets = {str(joint / name): sha256_file(joint / name) for name in frozen}
    mismatches = [name for name, expected in frozen.items() if assets[str(joint / name)] != expected]
    if mismatches:
        raise ValueError(f"pinned Joint V1 assets differ: {mismatches}")
    for seed, step in ((1, 40000), (2, 17004)):
        checkpoint = joint / f"checkpoints/aip_vtok/joint_v1/seed{seed}/step_{step}.pt"
        if checkpoint.exists():
            assets[str(checkpoint)] = sha256_file(checkpoint)
    historical = Path("results/original_adaptok_1080p")
    manifest = json.loads((historical / "checkpoint_manifest.json").read_bytes())
    directory = original / "checkpoints/original_adaptok_official" / manifest["revision"]
    for name, field in (("config.json", "config_sha256"), ("model.safetensors", "model_safetensors_sha256")):
        item = directory / name
        actual = sha256_file(item)
        if actual != manifest[field]:
            raise ValueError(f"pinned official asset differs: {name}")
        assets[str(item)] = actual
    for data in (*sorted(historical.glob("*manifest.json")), selection_path):
        assets[str(data.resolve())] = sha256_file(data)
    return assets


def hash_rgb(cohort, reader, output):
    rows = []
    for video, count in cohort.items():
        source = reader.catalog("validation")[video]
        for index in range(count):
            rgb = reader.read("validation", video, index).permute(1, 2, 0).contiguous()
            rows.append({"video": video, "index": index, "source_sha256": source["frame_hashes"][index]["sha256"],
                         "decoded_rgb24_sha256": hashlib.sha256(rgb.numpy().tobytes()).hexdigest()})
    write_csv(output / "validation_pixels.csv", rows)
    write_json(output / "validation_pixels_audit.json", {f"{k}_pixel_reads": v for k, v in reader.counts.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--joint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--hash-validation-rgb", action="store_true")
    args = parser.parse_args()
    repositories = {"original": repository(args.original), "joint": repository(args.joint),
                    "official_source": repository(args.original / "third_party/AdapTok")}
    assets = collect_assets(args.original, args.joint)
    record = {"repositories": repositories, "sha256": assets}
    if args.compare:
        previous = json.loads(args.compare.read_bytes())
        if previous != record:
            raise RuntimeError("tracked prior worktree state or archived asset hashes differ")
        record["verified_unchanged"] = True
    write_json(args.output / "assets.json", record)
    if args.hash_validation_rgb:
        path = Path("configs/hla_tok/validation_cohort.json")
        provenance = json.loads(path.read_bytes())
        reader = AuditedFrames(Path("data/manifests"), provenance["manifest_hashes"])
        hash_rgb(load_cohort(path, reader), reader, args.output)


if __name__ == "__main__":
    main()
