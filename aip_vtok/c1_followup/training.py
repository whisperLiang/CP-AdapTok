"""Resume the exact C1-A training sequence at its predeclared gate milestones."""

import hashlib
import json
import time
from dataclasses import dataclass

import torch
from torch import nn

from aip_vtok.c1.artifacts import timestamp
from aip_vtok.c1.recovery_net import RecoveryNet
from aip_vtok.c1.scan import DENSE
from aip_vtok.c1.training import LR, SAVE_EVERY, Run, Trainer, memory
from aip_vtok.diagnostics.diag0.io import json_at
from aip_vtok.runtime import file_hash, save_checkpoint, write_json
from aip_vtok.training.losses import ReconstructionLoss

from .calibration import legacy_inputs, sealed_training_plans
from .dense import Layout
from .experiment import PHASE1, frozen_candidates, load_layouts, load_phase1, validation_input
from .gates import LIMITS, sparse_release, training_gate
from .io import CANDIDATES, OUTPUT, PROTOCOL, WEIGHTS, write_rows
from .rollout import dense_rollout, recover, summarize_rollout

TRAINING_GATE = OUTPUT / "training_gate.json"


def recovery_model(groups, device):
    if groups not in (4, 6, 8):
        raise ValueError("dense RecoveryNet requires 4, 6 or 8 groups")
    model = RecoveryNet().to(device)
    if groups != 4:
        widen_inputs(model, groups)
    return model.to(memory_format=torch.channels_last)


@torch.no_grad()
def widen_inputs(model, groups):
    for kind, original in model.inputs.items():
        new = nn.Conv2d(original.in_channels + groups - 4, original.out_channels, 1,
                        device=original.weight.device, dtype=original.weight.dtype)
        new.weight.zero_()
        new.weight[:, :132] = original.weight[:, :132]
        new.weight[:, 128 + groups:] = original.weight[:, 132:]
        new.bias.copy_(original.bias)
        model.inputs[kind] = new


@dataclass(frozen=True)
class ReceivedDense:
    layout: Layout
    fit: object
    payload: tuple

    @property
    def space(self):
        return self.layout.space


class DenseTrainer(Trainer):
    """Use C1-A's loss, step, causal commits and detached references verbatim."""

    def __init__(self, codec, anchor, layouts, run, loss_weights):
        torch.manual_seed(run.seed)
        self.backbone, self.projector = codec.backbone, codec.projector
        self.model = recovery_model(layouts["I"].groups, "cuda")
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR, fused=True)
        self.objective = ReconstructionLoss(loss_weights).cuda().to(memory_format=torch.channels_last)
        self.anchor, self.layouts, self.run = anchor, layouts, run

    def receive(self, sample):
        layout = self.layouts[sample.frame_type]
        payload, _ = layout.encode(sample)
        return ReceivedDense(layout, self.anchor[sample.frame_type], payload)

    def latent(self, context, frame, budget):
        if budget != DENSE:
            raise ValueError("C1-A training is dense")
        return recover(frame.layout, self.backbone, frame.payload, self.model, context)


def plan_hash(plan):
    return hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@dataclass(frozen=True)
class SeedRun:
    arm: str
    seed: int
    identity: dict
    plan: list

    @property
    def weights(self):
        return WEIGHTS / self.arm / f"seed{self.seed}"

    @property
    def output(self):
        return OUTPUT / "dense_training" / self.arm / f"seed{self.seed}"

    def checkpoint(self, milestone):
        return self.weights / f"step_{milestone:04}.pt"

    def evidence(self, milestone):
        return self.output / f"step_{milestone:04}.json"


def job(arm, seed, plan):
    return SeedRun(arm, seed, {
        "arm": arm, "seed": seed, "plan_sha256": plan_hash(plan), "protocol_sha256": file_hash(PROTOCOL),
        "phase1_sha256": file_hash(PHASE1), "candidates_sha256": file_hash(CANDIDATES),
    }, plan)


def read_weights(path, identity):
    saved = torch.load(path, weights_only=True, map_location="cuda")
    if saved["identity"] != identity or len(saved["updates"]) != saved["step"]:
        raise RuntimeError(f"checkpoint training plan or step counts disagree at {path}")
    return saved


def restore_weights(trainer, seed_run, target):
    latest = seed_run.weights / "latest.pt"
    milestone = seed_run.checkpoint(target)
    path = milestone if milestone.is_file() else latest
    if not path.is_file():
        return []
    saved = read_weights(path, seed_run.identity)
    if saved["step"] > target:
        raise RuntimeError("checkpoint advanced past an unevaluated gate")
    trainer.model.load_state_dict(saved["model"])
    trainer.optimizer.load_state_dict(saved["optimizer"])
    return saved["updates"]


def save_progress(trainer, seed_run, updates, path):
    artifact = {
        "identity": seed_run.identity, "step": len(updates), "updates": updates,
        "model": trainer.model.state_dict(), "optimizer": trainer.optimizer.state_dict(),
    }
    save_checkpoint(path, artifact)
    write_json(seed_run.output / "updates.json", {"identity": seed_run.identity, "updates": updates})


def next_update(trainer, clip, plan, seed, index):
    tick = time.perf_counter()
    data = trainer.step(clip, plan["phase"], 100_000 * seed + index, verify_sync=True)
    return {"step": index, "seconds": time.perf_counter() - tick, "peak_allocated_mib": memory(), **data}


def train_to(run, train, seed_run, trainer, target):
    updates = restore_weights(trainer, seed_run, target)
    remaining = seed_run.plan[len(updates):target]
    torch.cuda.reset_peak_memory_stats()
    for plan, clip in zip(remaining, run.reader.training_windows(train, remaining), strict=True):
        index = len(updates) + 1
        update = next_update(trainer, clip, plan, seed_run.seed, index)
        updates.append(update)
        if index % SAVE_EVERY == 0:
            save_progress(trainer, seed_run, updates, seed_run.weights / "latest.pt")
            print(f"{seed_run.arm} seed={seed_run.seed} step={index}/{target} loss={update['loss']:.5f} "
                  f"grad={update['gradient_norm']:.5f}", flush=True)
    if len(updates) != target:
        raise RuntimeError("training ended before reaching its milestone")
    save_progress(trainer, seed_run, updates, seed_run.checkpoint(target))


def continuous_for(arm, layouts):
    arm_name = arm.replace("-fsq", "-cont") if arm.endswith("-fsq") else "A0"
    return layouts[arm_name]


def evaluate_milestone(run, seed, target, layouts, model):
    catalog, counts = validation_input(run)
    model.eval().requires_grad_(False)
    rows = []
    for video, count in counts.items():
        clip = run.reader.read(catalog, video, 0, count)
        rows.extend(dense_rollout(run.codec, clip, video, layouts, model, run.suite))
        print(f"{seed.arm} seed={seed.seed} step={target}: {video}, {count} Validation frames/arm", flush=True)
    table = summarize_rollout(rows, counts)
    gate = training_gate(table, target, LIMITS)
    integrity = run.verify_integrity()
    write_rows(seed.evidence(target).with_suffix(""), rows, identity=seed.identity, seed=seed.seed, step=target,
               table=table, gate=gate, integrity=integrity, cohort=counts,
               checkpoint_sha256=file_hash(seed.checkpoint(target)))
    print(f"{seed.arm} seed={seed.seed} step={target} gate: {gate}", flush=True)


def read_milestone(seed, target):
    record = json_at(seed.evidence(target))
    if record["identity"] != seed.identity or record["checkpoint_sha256"] != file_hash(seed.checkpoint(target)):
        raise RuntimeError("dense gate evidence refers to a different checkpoint or plan")
    table = summarize_rollout(record["rows"], record["cohort"])
    if record["table"] != table:
        raise RuntimeError("dense gate summary differs from its frame evidence")
    gate = training_gate(table, target, LIMITS)
    if record["step"] != target or record["seed"] != seed.seed or record["gate"] != gate:
        raise RuntimeError("dense gate outcome disagrees with its locked milestone")
    if not record["integrity"]["passed"] or record["integrity"]["test_images_read"] != 0:
        raise RuntimeError("dense gate lacks a valid integrity audit")
    return {key: value for key, value in record.items() if key != "rows"}


def run_milestone(run, train, anchor, layouts, seed, target):
    if not seed.evidence(target).is_file():
        trainer = DenseTrainer(run.codec, anchor, layouts["quantized"], Run("c1a", seed.seed, target, 4),
                               run.config["stage1"]["loss"])
        train_to(run, train, seed, trainer, target)
        evaluate_milestone(run, seed, target, layouts, trainer.model)
    return read_milestone(seed, target)


def run_candidate(run, train, anchor, layouts, jobs):
    seed1, seed2 = jobs
    pilot = run_milestone(run, train, anchor, layouts, seed1, LIMITS["pilot_steps"])
    if not pilot["gate"]["passed"]:
        return {"pilot": pilot, "seed1_final": None, "seed2_final": None,
                "steps": {"seed1": pilot["step"], "seed2": 0}, "sparse_released": False}
    final1 = run_milestone(run, train, anchor, layouts, seed1, LIMITS["final_steps"])
    outcome = {"pilot": pilot, "seed1_final": final1, "seed2_final": None,
               "steps": {"seed1": final1["step"], "seed2": 0}, "sparse_released": False}
    if final1["gate"]["passed"]:
        final2 = run_milestone(run, train, anchor, layouts, seed2, LIMITS["final_steps"])
        outcome["steps"]["seed2"] = final2["step"]
        outcome["seed2_final"] = final2
        outcome["sparse_released"] = sparse_release(True, final1, final2, run.verify_integrity(), LIMITS)
    return outcome


def run_training(run):
    admission = load_phase1()
    candidates = admission["released"]
    verdicts = {}
    if candidates:
        train = run.reader.open("train")
        plans = sealed_training_plans(train, LIMITS)
        _, anchor, _ = legacy_inputs(run.identity, "cuda")
        layouts = load_layouts(frozen_candidates(), "cuda")
        for arm in candidates:
            arm_layouts = {"continuous": continuous_for(arm, layouts), "quantized": layouts[arm]}
            jobs = [job(arm, seed, plan) for seed, plan in plans.items()]
            verdicts[arm] = run_candidate(run, train, anchor, arm_layouts, jobs)
    released = [arm for arm, value in verdicts.items() if value["sparse_released"]]
    decision = {"phase1_sha256": file_hash(PHASE1), "candidates": verdicts, "sparse_released": released,
                "integrity": run.verify_integrity(), "finished": timestamp()}
    write_json(TRAINING_GATE, decision)
    print(f"Phase 2/3 updates: {[(arm, r['steps']) for arm, r in verdicts.items()]}, sparse release {released}", flush=True)
    return decision
