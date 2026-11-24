"""Train matched, decoder-adapted spatial oracles on the allowed Train split."""

# ruff: noqa: E402
import argparse
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
from torch.nn import functional as F

from hla_tok import Geometry, HLATok
from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.bottleneck_diagnostics import DenseOracle
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import FullFrameQuality, sha256_file, summarize_quality
from hla_tok.evaluation.reconstruction import save_native
from hla_tok.training.checkpoint import ResumeIdentity, read_checkpoint, save_checkpoint
from hla_tok.training.train import TrainingFrames
from scripts.hla_tok.v2.protocol import (
    BASE,
    CHECKPOINTS,
    COHORT,
    OUTPUT,
    audit_read_counts,
    csv_at,
    frozen_data,
    heldout_examples,
    json_at,
    setup,
    verified_checkpoints,
)

STEPS = 500
CROP = Geometry(512, 512)
NATIVE = Geometry(1080, 1920)


def update(oracle, frame, optimizer):
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        image = oracle(frame, CROP)
        loss = F.l1_loss(image, frame) + 0.1 * F.mse_loss(image, frame)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(oracle.parameters(), 1.0, error_if_nonfinite=True)
    optimizer.step()
    return loss.item(), norm.item()


@torch.inference_mode()
def score(oracle, reader, quality, output):
    rows = []
    for item in heldout_examples():
        source = transfer(reader.read("validation", item["video"], item["index"])[None])
        with torch.autocast("cuda", dtype=torch.bfloat16):
            recon = oracle(source, NATIVE)
        rows.append({"video": item["video"], "index": item["index"]} | quality(source, recon))
        if item["video"].endswith("car-roundabout") and item["index"] == 32:
            save_native(output / "samples", "reconstruction", recon)
    write_csv(output / "per_frame.csv", rows)
    return summarize_quality(rows)


def execute(width, base, reader, quality, output):
    setup()
    codec = HLATok()
    codec.load_state_dict(base["model"], strict=True)
    oracle = DenseOracle(codec, width).cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW((p for p in oracle.parameters() if p.requires_grad), lr=2e-4, weight_decay=.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / 100))
    train = TrainingFrames(reader, [CROP] * STEPS, 117)
    history, start = [], perf_counter()
    for step, (raw, _, _, sample) in enumerate(train.batches(0), start=1):
        loss, norm = update(oracle, transfer(raw[None]), optimizer)
        scheduler.step()
        history.append({"step": step, "loss": loss, "grad_norm": norm, **sample})
    write_csv(output / "train.csv", history)
    meta = {"width": width, "dense_spatial_scalars_per_native_frame": 72 * 120 * width,
            "train_pixels": STEPS * CROP.width * CROP.height, "updates": STEPS,
            "wall_seconds": perf_counter() - start, "encoder_frozen": True, "rgb_decoder_adapted": True,
            "trainable_parameters": sum(p.numel() for p in oracle.parameters() if p.requires_grad),
            "objective": "RGB L1 + 0.1 RGB MSE", "no_discrete_rate": True}
    path = CHECKPOINTS / "diag0" / output.name / "step_00500.pt"
    identity = ResumeIdentity(f"v2/diag0/{output.name}", f"512:{STEPS}:seed117", sha256_file(COHORT), BASE)
    save_checkpoint(path, identity, oracle, optimizer, scheduler, torch.Generator(),
                    step=STEPS, data_cursor=STEPS, counters=meta)
    row = {"arm": output.name, **meta, "checkpoint_sha256": sha256_file(path)} | score(
        oracle.eval(), reader, quality, output)
    write_json(output / "measurement.json", row)
    print(output.name, row["global_psnr"], row["lpips"], row["dists"], flush=True)
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-root", required=True, type=Path)
    args = parser.parse_args()
    directory = OUTPUT / "diag0"
    query = json_at(directory / "query_summary.json")
    reader, _, identity = frozen_data()
    quality = FullFrameQuality("cuda", identity["metric_weights"])
    base = read_checkpoint(verified_checkpoints(args.legacy_root)["fixed"]).state
    oracles = directory / "oracles"
    oracles.mkdir(exist_ok=False)
    try:
        rows = [execute(width, base, reader, quality, oracles / arm) for arm, width in (("O0", 128), ("O1", 32))]
        write_csv(oracles / "metrics.csv", rows)
    finally:
        audit_read_counts(oracles, reader)
    attention = csv_at(directory / "attention_metrics.csv")
    fixed = [x for x in attention if x["stage"] == "fixed"]
    interventions = csv_at(directory / "intervention_summary.csv")
    write_json(directory / "diagnosis.json", {
        "status": "PASS", "legacy_reproductions": 3, "query": query,
        "fixed_query_attention_distinct_rank_cosine_mean": sum(float(x["different_rank_attention_cosine"])
                                                               for x in fixed) / len(fixed),
        "interventions": interventions, "oracles": rows,
        "attribution_scope": "The spatial encoder is frozen. The RGB decoder and bridge are adapted together. "
                             "Oracle results do not identify the local decoder in isolation.",
        "quantization_training_effect": "Requires a matched continuous-control trajectory in D3.",
    })


if __name__ == "__main__":
    main()
