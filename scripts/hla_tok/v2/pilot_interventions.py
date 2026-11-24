"""Apply identical token interventions to the trained continuous pilot arms."""

# ruff: noqa: E402
import argparse
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch

from hla_tok import HLATok
from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import FullFrameQuality, psnr, sha256_file
from hla_tok.training.v2 import load_experimental_checkpoint
from scripts.hla_tok.v2.diagnose import FIXED_BUDGET, attention_rows, intervention_rows
from scripts.hla_tok.v2.protocol import (
    COHORT,
    OUTPUT,
    ROOT,
    audit_read_counts,
    frozen_data,
    heldout_examples,
    json_at,
    setup,
)

CASES = ("original", "regional_mean", "repeat_rank0", "late_rank_replacement",
         "shuffle_values", "perturb_rank_assignment")


@torch.inference_mode()
def measure(seed, arms, output):
    setup(seed)
    output.mkdir(parents=True, exist_ok=False)
    reader, _, identity = frozen_data()
    quality = FullFrameQuality("cuda", identity["metric_weights"])
    groups = defaultdict(list)
    attention = []
    checkpoints = {}
    try:
        for arm in arms:
            model = HLATok(arm).cuda().to(memory_format=torch.channels_last).eval()
            folder = OUTPUT / f"continuous_pilot/seed{seed}/{arm}/pilot"
            path = ROOT / json_at(folder / "latest.json")["path"]
            state = load_experimental_checkpoint(path, model, sha256_file(COHORT)).state
            checkpoints[arm] = {"path": str(path.relative_to(ROOT)), "step": state["step"], "sha256": sha256_file(path)}
            for row in heldout_examples():
                source = transfer(reader.read("validation", row["video"], row["index"])[None])
                tags = {"arm": arm, "video": row["video"], "frame": row["index"]}
                if arm != "R1":
                    attention.extend(attention_rows(model, source, tags))
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    enc = model.encode_training(source, FIXED_BUDGET, quantize=False)
                    for variant in intervention_rows(model, source, enc, quality, tags, False, cases=CASES):
                        groups[arm, variant["intervention"]].append(variant)
            print("Trained pilot interventions", arm, seed, flush=True)
        write_json(output / "checkpoints.json", checkpoints)
        write_csv(output / "per_frame.csv", [r for group in groups.values() for r in group])
        summary = []
        for (arm, variant), data in groups.items():
            ref = groups[arm, "original"]
            summary.append({"arm": arm, "intervention": variant, "frames": len(data),
                            "global_psnr_delta": psnr(mean(d["mse"] for d in data)) - psnr(mean(d["mse"] for d in ref)),
                            **{key: mean(d[key] for d in data) for key in ("delta_lpips", "delta_dists", "delta_mse")}})
        write_csv(output / "metrics.csv", summary)
        if attention:
            write_csv(output / "attention.csv", attention)
    finally:
        audit_read_counts(output, reader)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--arms", nargs="+", choices=("R0", "R1", "R2", "R3"), default=("R0", "R1", "R2", "R3"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or OUTPUT / f"continuous_pilot/seed{args.seed}/interventions"
    measure(args.seed, args.arms, output)


if __name__ == "__main__":
    main()
