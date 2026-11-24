"""Read-only Joint V1 Validation provenance and archival rate evidence."""

import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

from hla_tok.evaluation.metrics import sha256_file


def archived_joint_rows(root: Path, identity: dict) -> list[dict]:
    """Bind matched archival results to the exact checkpoint and source frames."""
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    step = identity["joint_step"]
    checkpoint = root / f"checkpoints/aip_vtok/joint_v1/seed1/step_{step}.pt"
    work = root / f"results/aip_vtok/joint_v1/seed1/periodic/step_{step:05d}"
    frame_table = work / "validation_per_frame.csv"
    if commit != identity["joint_commit"] or sha256_file(checkpoint) != identity["joint_checkpoint_sha256"]:
        raise ValueError("unverified archived Joint V1 weights or repository")
    if sha256_file(frame_table) != identity["joint_step_40000_per_frame_sha256"]:
        raise ValueError("archived Joint V1 frame ledger differs from provenance")
    expected = {(r["video"], r["index"]): r["source_sha256"] for r in identity["frames"]}
    totals = defaultdict(list)
    with frame_table.open() as stream:
        for row in csv.DictReader(stream):
            if row["method"] == "Joint V1" and row["history"] == "CLOSED" and row["standard_cohort"] == "True":
                totals[int(row["budget"])].append(row)
    summary = json.loads((work / "evaluation.json").read_text())["summary"]
    result = []
    for row in summary:
        measured = totals[row["budget"]]
        obtained = {(r["video"], int(r["frame"])): r["input_rgb_sha256"] for r in measured}
        wire_bytes = sum(int(r["total_bytes"]) for r in measured)
        if obtained != expected or len(measured) != len(expected) or wire_bytes != row["total_bytes"]:
            raise ValueError("Joint V1 archival cohort or actual rate ledger differs")
        result.append(row | {"method": "AIP-VTok Joint V1", "measurement": "archived_matched_step40000",
                             "precision": "bf16", "token_per_frame": sum(int(r["transmitted_tokens"]) for r in measured)
                             / len(measured), "checkpoint_sha256": identity["joint_checkpoint_sha256"]})
    return result

