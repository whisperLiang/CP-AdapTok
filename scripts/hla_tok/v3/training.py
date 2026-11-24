"""One resumable train loop for dense, fixed and progressive reconstruction."""

import hashlib
import random
import subprocess
from dataclasses import dataclass
from math import cos, pi
from pathlib import Path
from time import perf_counter

import torch
from torch import nn

from hla_tok import BUDGETS, HLATok
from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.data import AuditedFrames
from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import FullFrameQuality, sha256_file
from hla_tok.models.anchors import anchor_digest
from hla_tok.models.spatial import SpatialEncoder
from hla_tok.training.checkpoint import ResumeIdentity, implementation_hash, read_checkpoint, save_checkpoint
from hla_tok.training.losses import ReconstructionLoss
from hla_tok.training.train import append_log, stopping_signal
from hla_tok.training.v3 import optimize, prefix_weights
from scripts.hla_tok.v2.protocol import COHORT, ROOT, json_at
from scripts.hla_tok.v3.protocol import (
    CHECKPOINTS,
    OUTPUT,
    PROTOCOL,
    Leg,
    SamplePlan,
    Trial,
    curriculum,
    encode_record,
)
from scripts.hla_tok.v3.quality import evaluate


def lr_factor(update: int, updates: int, warmup: int, floor: float) -> float:
    if update < warmup:
        return (update + 1) / warmup
    fraction = min(1.0, (update - warmup + 1) / max(1, updates - warmup))
    return floor + (1 - floor) * (1 + cos(pi * fraction)) / 2


def optimizer_for(model: nn.Module, leg: Leg, config: dict, previous: torch.optim.AdamW | None = None):
    if previous is None:
        optimizer = torch.optim.AdamW((param for param in model.parameters() if param.requires_grad),
                                      lr=leg.learning_rate, weight_decay=config["weight_decay"])
    else:
        optimizer = previous
        for group in optimizer.param_groups:
            group["lr"] = leg.learning_rate
            group["initial_lr"] = leg.learning_rate
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda update: lr_factor(update, leg.updates, config["warmup_updates"], config["minimum_lr_ratio"]))
    return optimizer, scheduler


def describe(model):
    if isinstance(model, HLATok):
        return model.architecture_metadata()
    return {"architecture_version": "hla-tok.dense", "rgb_decoder": model.rgb_decoder.variant}


def configure(model, trial, leg):
    model.encoder.requires_grad_(not leg.freeze_spatial)
    model.rgb_decoder.requires_grad_(not leg.freeze_spatial)
    if isinstance(model, HLATok):
        model.quantizer.requires_grad_(trial.quantized)


def _identity(trial, loss, provenance):
    digest = hashlib.sha256(implementation_hash(ROOT).encode())
    for name in ("training.py", "protocol.py", "quality.py", "calibration.py"):
        digest.update(Path(__file__).with_name(name).read_bytes())
    cfg = hashlib.sha256(PROTOCOL.read_bytes() + encode_record({"weights": loss.weights, "init": provenance}))
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT).strip()
    return ResumeIdentity(f"v3/{trial.relative}", cfg.hexdigest(), sha256_file(COHORT), commit, digest.hexdigest())


def check_model_metadata(model, state):
    expected = describe(model)
    if state["counters"]["architecture"] != expected:
        raise ValueError("checkpoint architecture differs")
    anchors = state["model"].get("query_encoder.anchor_table")
    if anchors is not None and anchor_digest(anchors) != expected["anchor_table_sha256"]:
        raise ValueError("checkpoint anchor table SHA-256 differs")


def _teacher_weight(config, update, teacher):
    if teacher is None:
        return 0.0
    latent = config["latent"]
    fraction = (latent["decay_end"] - update) / (latent["decay_end"] - latent["decay_start"])
    return latent["weight"] * min(1.0, max(0.0, fraction))


@dataclass(frozen=True)
class TrialRun:
    trial: Trial
    config: dict
    model: nn.Module
    loss: ReconstructionLoss
    teacher: SpatialEncoder | None
    provenance: dict
    reader: AuditedFrames
    cohort: dict
    quality: FullFrameQuality

    @property
    def output(self):
        return OUTPUT / self.trial.relative

    @property
    def checkpoints(self):
        return CHECKPOINTS / self.trial.relative

    def observation(self, step: int, *, full: bool):
        target = self.output / ("full" if full else "trend") / f"step_{step:05d}"
        if (target / "metrics.json").is_file():
            return json_at(target / "metrics.json")
        budgets: tuple[int, ...] = () if self.trial.stage == "dense" else (self.config["native_budget"],)
        if self.trial.stage == "progressive" and full:
            budgets = BUDGETS
        rows = evaluate(self.model, self.reader, self.cohort, self.quality, target, budgets=budgets,
                        quantized=self.trial.quantized, subset=not full, bypass=self.trial.arm == "C1")
        for row in rows:
            print(f"{self.trial.arm} seed{self.trial.seed} step={step} frames={row['frames']} "
                  f"{row['mode']} K{row['budget']} PSNR={row['global_psnr']:.5f} "
                  f"LPIPS={row['lpips']:.5f} DISTS={row['dists']:.5f}", flush=True)
        return rows

    def due(self, step: int, base: int, limit: int):
        if step > 0 and step % self.config["checkpoint_every"] == 0:
            self.observation(step, full=False)
        if step > 0 and step in {*self.config["full_validation_updates"], base, limit}:
            self.observation(step, full=True)

    def update(self, batch, optimizer, step):
        raw, geometry, _, sample = batch
        fixed = self.config["native_budget"]
        beta = self.config["progressive"]["fixed_weight"] if self.trial.stage == "progressive" else 0.0
        budgets = ({} if self.trial.stage == "dense" else
                   prefix_weights(geometry, sample["native_budget"], fixed, beta))
        self.model.train()
        return optimize(self.model, transfer(raw[None]), self.loss, sample["perceptual_crops_yx"],
                        budgets, optimizer, quantize=self.trial.quantized, teacher=self.teacher,
                        latent_weight=_teacher_weight(self.config, step, self.teacher),
                        clip_norm=self.config["gradient_clip_norm"])


def latest_path(run):
    return max(run.checkpoints.glob("step_*.pt"), default=None)


def _restore(run, identity, plan, legs, rng):
    source = latest_path(run)
    if source is None:
        counters = {"architecture": describe(run.model), "sample_chain": plan.chains[0], "seen_pixels": 0,
                    "optimizer_updates": 0, "optimization_seconds": 0.0, "budget_updates": {}, "leg": 0}
        return counters, None, None
    checkpoint = read_checkpoint(source, identity)
    info = checkpoint.state
    counters, step = info["counters"], info["step"]
    if counters["optimizer_updates"] != step or counters["sample_chain"] != plan.chains[step]:
        raise ValueError("checkpoint update count or sample chain disagrees with the frozen plan")
    check_model_metadata(run.model, info)
    leg = legs[counters["leg"]]
    configure(run.model, run.trial, leg)
    optimizer, scheduler = optimizer_for(run.model, leg, run.config)
    checkpoint.restore(run.model, optimizer, scheduler, rng)
    return counters, optimizer, scheduler


def _progress(run, counters, *, stopped=False):
    write_json(run.output / "training.json", counters | {
        "physical_batch": 1, "effective_batch": 1, "stopping_requested": stopped,
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    })


def _snapshot(run, identity, counters, optimizer, scheduler, rng):
    step = counters["optimizer_updates"]
    path = run.checkpoints / f"step_{step:05d}.pt"
    if not path.is_file():
        save_checkpoint(path, identity, run.model, optimizer, scheduler, rng,
                        step=step, data_cursor=step, counters=counters)
    write_json(run.output / "latest.json", {"step": step, "path": str(path), "sha256": sha256_file(path)})


def _update_span(run, start, end, plan, counters, optimizer, scheduler, rng, identity, base, limit, stop):
    log = run.output / f"updates_after_{start:05d}.jsonl"
    if log.exists():
        log = run.output / f"updates_after_{start:05d}_{perf_counter():.6f}.jsonl"
    for step, batch in enumerate(plan.batches(start, end), start + 1):
        if stop():
            break
        begin = perf_counter()
        lr = optimizer.param_groups[0]["lr"]
        row = run.update(batch, optimizer, step)
        scheduler.step()
        seconds = perf_counter() - begin
        geometry, sample = batch[1], batch[-1]
        counters["sample_chain"] = plan.chains[step]
        counters["seen_pixels"] += geometry.height * geometry.width
        counters["optimizer_updates"] = step
        counters["optimization_seconds"] += seconds
        if run.trial.stage != "dense":
            budget = str(sample["native_budget"])
            counters["budget_updates"][budget] = counters["budget_updates"].get(budget, 0) + 1
        append_log(log, {"step": step, "seconds": seconds, "lr": lr, "sample": sample, **row})
        if step % 25 == 0:
            _progress(run, counters)
        if step % run.config["checkpoint_every"] == 0 or step == end or stop():
            _snapshot(run, identity, counters, optimizer, scheduler, rng)
        if not stop():
            run.due(step, base, limit)
    completed = counters["optimizer_updates"]
    if stop() and completed > start:
        _snapshot(run, identity, counters, optimizer, scheduler, rng)


def execute(run: TrialRun, *, extend: bool = False):
    run.output.mkdir(parents=True, exist_ok=True)
    random.seed(run.trial.seed)
    torch.manual_seed(run.trial.seed)
    identity = _identity(run.trial, run.loss, run.provenance)
    legs = curriculum(run.config, run.trial.stage)
    normal = [leg.geometry for leg in legs for _ in range(leg.updates)]
    extension = run.config["extension"]["max_updates"]
    limit = len(normal) + (extension if extend else 0)
    plan = SamplePlan(run.reader, normal + [normal[-1]] * extension, run.trial.seed, run.config,
                      progressive=run.trial.stage == "progressive")
    plan.save(OUTPUT / "plans" / f"{run.trial.stage}_seed{run.trial.seed}.jsonl")
    budget_rng = torch.Generator().manual_seed(run.trial.seed)
    counters, optimizer, scheduler = _restore(run, identity, plan, legs, budget_rng)
    initial_record = run.provenance | {"planned_sample_chain": plan.chains[len(normal)],
                                      "extension_sample_chain": plan.chains[-1], "loss_weights": run.loss.weights,
                                      "trainable_master_dtype": str(next(run.model.parameters()).dtype)}
    write_json(run.output / "initialization.json", initial_record)
    first = counters["optimizer_updates"]
    if first > limit:
        raise ValueError("checkpoint extends beyond the authorized run")
    count_before = dict(run.reader.counts)
    torch.cuda.reset_peak_memory_stats()
    with stopping_signal() as stop:
        try:
            run.due(first, len(normal), limit)
            if first == 0 and run.trial.arm == "C1":
                run.observation(0, full=True)
            end = 0
            for index, leg in enumerate(legs):
                offset, end = end, end + leg.updates
                stop_at = limit if index == len(legs) - 1 else end
                if max(offset, first) >= stop_at or stop():
                    continue
                configure(run.model, run.trial, leg)
                if optimizer is None or counters["leg"] != index:
                    retain = index > 0 and legs[index - 1].freeze_spatial == leg.freeze_spatial
                    optimizer, scheduler = optimizer_for(run.model, leg, run.config, optimizer if retain else None)
                counters["leg"] = index
                _update_span(run, max(first, offset), stop_at, plan, counters, optimizer, scheduler, budget_rng,
                             identity, len(normal), limit, stop)
        finally:
            _progress(run, counters, stopped=stop())
            counts = {key: run.reader.counts[key] - count_before[key] for key in count_before if key != "validation"}
            prior = run.output / f"train_reads_after_{first:05d}.jsonl"
            append_log(prior, counts)
        completed = counters["optimizer_updates"]
        if stop():
            return None
    if completed != limit:
        raise RuntimeError("run ended without completing its declared update budget")
    return run.observation(limit, full=True)
