"""Re-score archived Joint V1 Validation bytes with the same HLA-Tok RGB metrics."""

import argparse
import csv
import json
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

# ruff: noqa: E402
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import torch

from hla_tok.evaluation.data import AuditedFrames, load_cohort
from hla_tok.evaluation.io import write_csv, write_json


def archived_frames(path):
    table = {}
    with path.open() as stream:
        for row in csv.DictReader(stream):
            if row["method"] == "Joint V1" and row["history"] == "CLOSED" and row["standard_cohort"] == "True":
                table[row["video"], int(row["frame"]), int(row["budget"])] = row
    return table


def replay_video(codec, block, reader, video, count, archived, quality, streams, budgets):
    from aip_vtok.decoder.causal_state import ReferenceState
    from aip_vtok.joint.evaluation import rms, transmitted_tokens

    profile, reference = codec.profile(), ReferenceState()
    rows = defaultdict(list)
    for index in range(count):
        wire = (streams / f"{index:04d}.aipj").read_bytes()
        session = codec.begin_frame(reference, wire[:block], profile)
        packets = iter(profile.wire.packets(wire))
        size = block
        target = reader.read("validation", video, index)[None].cuda().float().div_(255)
        for budget in budgets:
            required = transmitted_tokens(budget, session.context.intra)
            while session.received < required:
                packet = next(packets)
                session.append_packet(packet)
                size += len(packet)
                if session.received == session.reference_budget:
                    reference = session.commit_reference()
            record = archived[video, index, budget]
            if size != int(record["total_bytes"]):
                raise RuntimeError(f"archived byte count mismatch at {video} {index} K{budget}")
            if budget == budgets[0]:
                expected = record["rx_reference_sha256"]
                if expected and reference.fingerprint() != expected:
                    raise RuntimeError(f"archived reference fingerprint mismatch at {video} {index}")
                for name in ("latent", "feature"):
                    value = rms(getattr(reference, name))
                    target_rms = float(record[f"reference_{name}_rms"])
                    if not math.isclose(value, target_rms, rel_tol=1e-5, abs_tol=1e-9):
                        raise RuntimeError(f"archived reference {name} RMS differs at {video} {index}")
            scored = quality(target, session.render(budget))
            if not math.isclose(scored["mse"], float(record["mse"]), rel_tol=1e-5, abs_tol=1e-9):
                raise RuntimeError(f"archived reconstruction MSE differs at {video} {index} K{budget}")
            rows[budget].append({"video": video, "frame": index, "budget": budget, "total_bytes": size} | scored)
    return rows


@torch.no_grad()
def evaluate(joint, output):
    sys.path.insert(0, str(joint))
    from aip_vtok.joint import protocol
    from aip_vtok.joint.bitstream import FRAME_HEADER
    from aip_vtok.joint.evaluation import BUDGETS, load_selection

    from hla_tok.evaluation.joint_evidence import archived_joint_rows
    from hla_tok.evaluation.metrics import FullFrameQuality, sha256_file, summarize_quality

    output.mkdir(parents=True, exist_ok=False)
    identity = json.loads((ROOT / "configs/hla_tok/validation_cohort.json").read_bytes())
    comparison = {r["budget"]: r for r in archived_joint_rows(joint, identity)}
    raw = AuditedFrames(joint / "data/manifests", identity["manifest_hashes"])
    cohort = load_cohort(ROOT / "configs/hla_tok/validation_cohort.json", raw)
    archived_dir = joint / "results/aip_vtok/joint_v1/seed1/validation"
    frame_table = archived_dir / "validation_per_frame.csv"
    if sha256_file(frame_table) != identity["joint_per_frame_sha256"]:
        raise ValueError("official Validation rate ledger differs from pinned evidence")
    archived = archived_frames(frame_table)
    # Redirect the old loader's generated migration report into this experiment.
    protocol.OUTPUT = output
    run = protocol.open_experiment(joint / protocol.CONFIG, seed=1)
    selection = load_selection(run, seed=1)["chosen"]
    if selection["step"] != identity["joint_step"] or selection["checkpoint_sha256"] != identity["joint_checkpoint_sha256"]:
        raise ValueError("selected checkpoint differs from the pinned step 40000 comparison")
    quality = FullFrameQuality("cuda:0", identity["metric_weights"])
    result = defaultdict(list)
    try:
        for video, count in cohort.items():
            streams = archived_dir / "streams" / video.replace(":", "_") / "CLOSED"
            scored = replay_video(run.codec, FRAME_HEADER.size, raw, video, count, archived, quality, streams, BUDGETS)
            for budget, frames in scored.items():
                result[budget].extend(frames)
            write_csv(output / "per_frame.csv", [row for budget in BUDGETS for row in result[budget]])
            print(f"Joint V1 CLOSED replay verified {video} {count} frames", flush=True)
        run.verify_integrity()
        rows = [comparison[k] | summarize_quality(result[k]) |
                {"measurement": "verified_archived_bytes_native_rgb_replay"} for k in BUDGETS]
        write_csv(output / "metrics.csv", rows)
    finally:
        write_json(output / "pixel_reads.json", {f"{key}_pixel_reads": value for key, value in raw.counts.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    joint = args.joint_root.resolve()
    expected = json.loads((ROOT / "configs/hla_tok/validation_cohort.json").read_bytes())["joint_commit"]
    commit = subprocess.check_output(["git", "-C", str(joint), "rev-parse", "HEAD"], text=True).strip()
    if commit != expected:
        raise ValueError("Joint V1 replay needs the pinned read-only worktree")
    evaluate(joint, args.output.resolve())


if __name__ == "__main__":
    main()
