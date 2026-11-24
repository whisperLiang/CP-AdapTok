"""One training entry point for the gated V2 arms and matched controls."""

# ruff: noqa: E402
import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import torch
import yaml
from PIL import Image

from hla_tok import Geometry
from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import FullFrameQuality, psnr, sha256_file
from hla_tok.evaluation.reconstruction import validate
from hla_tok.evaluation.representation_diagnostics import summarize_prefix
from hla_tok.training.checkpoint import ResumeIdentity, implementation_hash, read_checkpoint, save_checkpoint
from hla_tok.training.losses import Losses
from hla_tok.training.step import optimize_batch
from hla_tok.training.train import TrainingFrames, append_log, calibrate, stopping_signal
from hla_tok.training.v2 import load_experimental_checkpoint, matched_initialization
from scripts.hla_tok.v2.protocol import (
    CHECKPOINTS,
    COHORT,
    CONFIG,
    OUTPUT,
    ROOT,
    audit_read_counts,
    csv_at,
    frozen_data,
    json_at,
    setup,
    verified_checkpoints,
)

STAGES = {"rgb": "diag1_overfit", "overfit": "diag1_overfit", "pilot": "continuous_pilot",
          "quantized": "quantized", "control": "quantized", "native": "native", "continue": "native"}
ARMS = ("R0", "R1", "R2", "R3")
REP_FIELDS = ("effective_rank", "stable_rank", "centered_variance", "common_energy_fraction",
              "raw_adjacent_cosine", "raw_distinct_cosine", "centered_distinct_cosine")


@dataclass(frozen=True)
class Trial:
    stage: str
    arm: str
    seed: int
    updates: int
    every: int
    geometry: Geometry
    fixed_samples: int
    quantized: bool

    @property
    def relative(self):
        group = STAGES[self.stage]
        if group == "diag1_overfit":
            return Path(group) / self.arm / self.stage
        return Path(group) / f"seed{self.seed}" / self.arm / self.stage

    @property
    def output(self):
        return OUTPUT / self.relative

    @property
    def checkpoints(self):
        return CHECKPOINTS / self.relative


class RepresentationLog:
    def __init__(self):
        self.rows = []

    def __call__(self, encoded, video, frame):
        stats = summarize_prefix(encoded.continuous, encoded.geometry.num_regions)
        self.rows.append({"video": video, "frame": frame,
                          **{key: stats[key].mean().item() for key in REP_FIELDS}})

    def summary(self):
        return {key: mean(row[key] for row in self.rows) for key in REP_FIELDS}


def trials(args, config):
    stage = config[args.stage]
    geometry = Geometry(stage["height"], stage["width"])
    arms = ARMS if args.arm == "all" else (args.arm,)
    for arm in arms:
        yield Trial(args.stage, arm, args.seed, stage["updates"] + (1000 if args.extend else 0),
                    stage["every"], geometry, stage["fixed_samples"], stage["quantized"])


def plan_digest(sample):
    return json.dumps(sample, sort_keys=True, separators=(",", ":")).encode()


def fixed_examples(reader, count, geometry):
    samples = []
    for item in json_at(CONFIG / "diagnostic_frames.json")["train_overfit"][:count]:
        record = reader.catalog("train")[item["video"]]
        if record["frame_hashes"][item["index"]]["sha256"] != item["source_sha256"]:
            raise ValueError("frozen Train crop has a different source SHA-256")
        frame = reader.read("train", item["video"], item["index"])
        x, y, w, h = item["crop_xywh"]
        samples.append((transfer(frame[None, :, y:y+h, x:x+w]), geometry, (0, 0), item))
    return samples


def input_batches(reader, trial, first, samples):
    if samples:
        for update in range(first, trial.updates):
            yield samples[update % len(samples)]
    else:
        loader = TrainingFrames(reader, [trial.geometry] * trial.updates, trial.seed)
        for raw, geometry, crop, sample in loader.batches(first):
            yield transfer(raw[None]), geometry, crop, sample


@torch.inference_mode()
def score_crops(model, cases, quality, count, output):
    model.eval()
    reps = RepresentationLog()
    rows = []
    for source, _, _, record in cases:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            encoded = model.encode_training(source, count, quantize=False)
            image = model.decode_training(encoded, count)
        metrics = quality.suite(source, image)
        metrics["lpips_vgg"] = quality.vgg(source * 2 - 1, image * 2 - 1).mean().item()
        rows.append(metrics | record)
        reps(encoded, record["video"], record["index"])
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            label = record["video"].replace(":", "_")
            for name, value in (("source", source), ("reconstruction", image)):
                rgb = value[0].mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
                Image.fromarray(rgb).save(output / f"{label}_{name}.png")
    mse = mean(row["mse"] for row in rows)
    return {"frames": len(rows), "global_psnr": psnr(mse), "mse": mse,
            **{k: mean(r[k] for r in rows) for k in ("lpips", "lpips_vgg", "dists")},
            **reps.summary()}


def native_score(model, reader, cohort, quality, output, *, quantized, subset):
    reps = RepresentationLog()
    row, = validate(model, reader, cohort, quality, (16320,), output,
                    quantize=quantized, subset=subset, observe=reps)
    write_csv(output / "representation.csv", reps.rows)
    result = row | reps.summary()
    write_json(output / "summary.json", result)
    return result


def identity_for(trial):
    program = hashlib.sha256()
    program.update(implementation_hash(ROOT).encode())
    program.update(Path(__file__).read_bytes())
    return ResumeIdentity(f"v2/{trial.stage}/{trial.arm}/{trial.seed}", sha256_file(CONFIG / "protocol.yaml"),
                          sha256_file(COHORT), subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                          program.hexdigest())


def initialize(model, trial, identity, args, optimizer, scheduler, rng):
    if args.resume:
        checked = load_experimental_checkpoint(args.resume, model, identity.dataset_sha256)
        loaded = read_checkpoint(args.resume, identity)
        loaded.restore(model, optimizer, scheduler, rng)
        return checked.state["step"], checked.state["counters"]
    if args.initialize_from:
        load_experimental_checkpoint(args.initialize_from, model, identity.dataset_sha256)
    if trial.stage == "continue":
        old = read_checkpoint(verified_checkpoints(args.legacy_root)["fixed"]).state
        model.load_state_dict(old["model"], strict=True)
        optimizer.load_state_dict(old["optimizer"])
    return 0, {"architecture": model.architecture_metadata(), "seen_frames": 0, "seen_pixels": 0,
               "optimization_seconds": 0.0, "validation_seconds": 0.0, "sample_chain": bytes(32).hex()}


def snapshot(model, trial, identity, optimizer, scheduler, rng, counters, step):
    path = trial.checkpoints / f"step_{step:05d}.pt"
    save_checkpoint(path, identity, model, optimizer, scheduler, rng, step=step, data_cursor=step, counters=counters)
    write_json(trial.output / "latest.json", {"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path),
                                              "step": step})
    return path


def train_step(model, trial, current, optimizer, scheduler, losses, config, counters, *, audit, sample):
    frame, geometry, perceptual, _ = current
    start = perf_counter()
    row = optimize_batch(model, frame, (geometry.scale_budget(config["native_budget"]),), optimizer, losses,
                         quantize=trial.quantized, bf16=True, audit=audit, crop_position=perceptual,
                         clip_norm=config["gradient_clip_norm"])
    scheduler.step()
    counters["seen_frames"] += 1
    counters["seen_pixels"] += geometry.height * geometry.width
    counters["optimization_seconds"] += perf_counter() - start
    counters["sample_chain"] = hashlib.sha256(bytes.fromhex(counters["sample_chain"]) + plan_digest(sample)).hexdigest()
    counters["latest_codebook"] = row["codebook"]
    return row


def periodic(model, trial, reader, cohort, quality, cases, step):
    target = trial.output / "periodic" / f"step_{step:05d}"
    if cases:
        images = target / "samples" if step == trial.updates else None
        return score_crops(model, cases, quality, trial.geometry.scale_budget(16320), images)
    row = native_score(model, reader, cohort, quality, target, quantized=trial.quantized, subset=True)
    if trial.stage == "quantized":
        bypass = native_score(model, reader, cohort, quality, target / "bypass", quantized=False, subset=True)
        row |= {f"bypass_{key}": bypass[key] for key in ("global_psnr", "lpips", "dists", *REP_FIELDS)}
    return row


def output_progress(trial, counters, model, initial, completed, start):
    motion = {name: (tensor.detach().cpu().float() - initial[name]).square().sum()
              for name, tensor in model.named_parameters() if name in initial}
    write_json(trial.output / "training.json", counters | {
        "optimizer_updates": completed, "physical_batch": 1, "effective_batch": 1,
        "segment_seconds": perf_counter() - start, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
        "parameter_update_l2": sum(x.item() for x in motion.values())**0.5,
    })


def execute(trial, config, args, quality, perceptual):
    setup(trial.seed)
    start = perf_counter()
    if args.resume:
        if not trial.output.is_dir():
            raise ValueError("resume requires the existing V2 trial directory")
    else:
        trial.output.mkdir(parents=True, exist_ok=False)
    reader, cohort, _ = frozen_data()
    model, init = matched_initialization(trial.arm, trial.seed)
    model = model.cuda().to(memory_format=torch.channels_last)
    identity = identity_for(trial)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / config["warmup_steps"]))
    rng = torch.Generator().manual_seed(trial.seed)
    first, counters = initialize(model, trial, identity, args, optimizer, scheduler, rng)
    initial = {name: x.detach().cpu().float().clone() for name, x in model.named_parameters()}
    cases = fixed_examples(reader, trial.fixed_samples, trial.geometry)
    losses = Losses("cuda", lpips_vgg=0, dists=0) if trial.stage == "rgb" else perceptual
    if not args.resume:
        if trial.stage == "quantized":
            calibrate(model, reader, config["calibration_frames"], config["native_budget"], trial.seed, trial.output)
        write_json(trial.output / "initialization.json", init)
    history_path = trial.output / "training_curve.csv"
    history = csv_at(history_path) if args.resume and history_path.is_file() else []
    log_path = trial.output / f"updates_after_{first:05d}.jsonl"
    log_path.touch(exist_ok=False)
    if cases:
        history.append({"step": 0} | periodic(model, trial, reader, cohort, quality, cases, 0))
    torch.cuda.reset_peak_memory_stats()
    completed = first
    with stopping_signal() as stop:
        try:
            for step, batch in enumerate(input_batches(reader, trial, first, cases), start=first + 1):
                model.train()
                row = train_step(model, trial, batch, optimizer, scheduler, losses, config, counters,
                                 audit=step == first + 1, sample=batch[-1])
                row |= {"step": step, "sample": batch[-1], "lr": optimizer.param_groups[0]["lr"]}
                append_log(log_path, row)
                completed = step
                if step % trial.every == 0 or stop():
                    snapshot(model, trial, identity, optimizer, scheduler, rng, counters, step)
                if stop():
                    break
                if step % trial.every == 0:
                    begun = perf_counter()
                    observed = periodic(model, trial, reader, cohort, quality, cases, step)
                    counters["validation_seconds"] += perf_counter() - begun
                    history.append({"step": step} | observed)
                    write_csv(history_path, history)
                    print(trial.stage, trial.arm, trial.seed, step, observed["global_psnr"], observed["lpips"],
                          observed["dists"], flush=True)
            if completed == trial.updates and not cases:
                begun = perf_counter()
                native_score(model, reader, cohort, quality, trial.output / "validation", quantized=trial.quantized,
                             subset=False)
                if trial.stage in {"quantized", "native"}:
                    native_score(model, reader, cohort, quality, trial.output / "bypass_validation", quantized=False,
                                 subset=False)
                counters["validation_seconds"] += perf_counter() - begun
        finally:
            output_progress(trial, counters, model, initial, completed, start)
            audit_read_counts(trial.output / f"reads_after_{first:05d}", reader)


def gate(args):
    if args.stage in {"rgb", "overfit"}:
        path = OUTPUT / "diag0/diagnosis.json"
    elif args.stage == "pilot":
        path = OUTPUT / "diag1_overfit/acceptance.json"
    elif args.stage in {"quantized", "control"}:
        path = OUTPUT / "continuous_pilot/selection.json"
    else:
        path = OUTPUT / "quantized/acceptance.json"
    if not path.is_file() or json_at(path)["status"] != "PASS":
        raise RuntimeError(f"prior V2 phase has no full pass: {path}")
    if args.stage in {"quantized", "control", "native"} and not (args.initialize_from or args.resume):
        raise ValueError("this phase needs the selected prior checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--arm", choices=(*ARMS, "all"), default="all")
    parser.add_argument("--seed", type=int, choices=(1, 2), default=1)
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--legacy-root", type=Path)
    parser.add_argument("--extend", action="store_true")
    args = parser.parse_args()
    if args.extend and (args.stage != "pilot" or args.resume is None):
        raise ValueError("the pilot extension resumes its original optimizer and cursor")
    gate(args)
    setup(args.seed)
    config = yaml.safe_load((CONFIG / "protocol.yaml").read_bytes())
    identity = json_at(COHORT)
    quality = FullFrameQuality("cuda", identity["metric_weights"])
    perceptual = Losses("cuda", **config["loss"])
    for trial in trials(args, config):
        execute(trial, config, args, quality, perceptual)


if __name__ == "__main__":
    main()
