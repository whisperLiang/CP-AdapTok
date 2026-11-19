"""Native-resolution truncated-gradient training in the receiver's causal loop."""

import math
import statistics
import time
from dataclasses import asdict, dataclass, replace
from itertools import chain

import numpy as np
import torch

from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.diagnostics.diag0.closed import same_state
from aip_vtok.runtime import file_hash, save_checkpoint, write_json
from aip_vtok.tokenizer.fsq import FSQ
from aip_vtok.training.data import prefetched_windows
from aip_vtok.training.losses import ReconstructionLoss, crop_plan

from .artifacts import LOCKED, OUTPUT, WEIGHTS
from .calibrate import key, sparse_latent, sparse_ordered
from .channels import ALIGNED, ChannelFit
from .diagnostic import p_space
from .frames import capture
from .recovery_net import RecoveryNet, commit_reference, reference_supervision
from .scan import BUDGETS, DENSE, REF_BUDGET, SHAPE, TOP
from .tokens import Observations, dense_roundtrip, receive, send

ANCHOR_WEIGHT = 0.05
REFERENCE_WEIGHT = 0.15
LR = 1e-3
SAVE_EVERY = 50
SMOKE_UPDATES = 10
SEEDS = (1, 2)


@dataclass(frozen=True)
class Run:
    phase: str
    seed: int
    steps: int
    length: int
    late_gop: bool = False

    @property
    def sparse(self):
        return self.phase == "c1b"

    @property
    def rollout(self):
        return f"rollout_{self.length}f_detached_ref"


@dataclass(frozen=True)
class ReceivedFrame:
    """A receiver has only the static protocol and the current FSQ payload."""

    fit: ChannelFit
    space: str
    observations: Observations | None
    fsq: FSQ
    slots: torch.Tensor | None = None
    payload: torch.Tensor | None = None

    def read(self, budget):
        if budget == DENSE and self.observations is not None:
            return self.observations
        return receive(self.fsq, self.payload[:, :budget], self.fit.scales[:4],
                       self.slots[:budget], SHAPE, "fsq")


def dense_observations(fsq, sample, fit, space, *, quantized=True):
    values = sparse_ordered(sample, space, fit)
    if quantized:
        values, _ = dense_roundtrip(fsq, values, fit.scales[:4])
    return Observations(values, torch.ones_like(values[:, :, 0], dtype=torch.bool))


def prepare_frame(sample, fits, lock, fsq, slots):
    space = p_space(sample, lock)
    fit = fits[key(sample.frame_type, space)]
    if slots is None:
        observations = dense_observations(fsq, sample, fit, space)
        return ReceivedFrame(fit, space, observations, fsq)
    packet = send(fsq, sparse_ordered(sample, space, fit), fit.scales[:4], slots)
    return ReceivedFrame(fit, space, None, fsq, slots, packet.payload)


def predict(backbone, model, obs, fit, space, sigma, context):
    base = sparse_latent(backbone, fit, space, obs, sigma)
    return base if model is None else model(base, obs.mask, fit.indices[:TOP], context)


def budgets_for(context, sparse, rng):
    if not sparse:
        return (DENSE,)
    ref = REF_BUDGET[context.intra]
    middle = int(rng.choice([k for k in BUDGETS[1:-1] if k > ref]))
    return ref, middle, BUDGETS[-1]


def weighted_anchor(aligned, target, fit, backbone, space):
    std = fit.std.view(1, -1, 1, 1)
    if space != ALIGNED:
        std = std / backbone.alignment.p_std
    weights = fit.weights.view(1, -1, 1, 1)
    return (weights * ((aligned - target) / std.clamp_min(1e-3)).abs()).mean()


def make_plans(catalog, run):
    if catalog.split != "train":
        raise ValueError("C1 optimization requires Train")
    videos = sorted(catalog.rows)
    rng = np.random.default_rng(run.seed)
    epochs = math.ceil(run.steps / len(videos))
    shuffled = chain.from_iterable(rng.permutation(len(videos)).tolist() for _ in range(epochs))
    plans = []
    for step, index in enumerate(shuffled):
        if step == run.steps:
            break
        length = catalog.rows[videos[index]]["original_frames"]
        # Include early, middle and late phases. Start a real causal state at the
        # selected GOP origin and warm it up with its actual preceding frames.
        choices = (0, 0, 0, 0, 4, 12, 20, 26) if run.late_gop else (0,)
        phase = int(rng.choice([p for p in choices if p + run.length <= length]))
        span = phase + run.length
        plans.append({"source": index, "start": int(rng.integers(length - span + 1)),
                      "length": span, "phase": phase})
    return plans


class Trainer:
    def __init__(self, codec, fits, lock, run, loss_weights):
        torch.manual_seed(run.seed)
        self.backbone, self.projector = codec.backbone, codec.projector
        self.model = RecoveryNet().cuda().to(memory_format=torch.channels_last)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR, fused=True)
        self.objective = ReconstructionLoss(loss_weights).cuda().to(memory_format=torch.channels_last)
        self.fsq = FSQ().cuda()
        self.fits, self.lock, self.run = fits, lock, run
        self.slots = torch.tensor(lock["slots"], device="cuda") if run.sparse else None
        self.sigma = lock["winner"]["sigma"] if run.sparse else None

    def receive(self, sample):
        return prepare_frame(sample, self.fits, self.lock, self.fsq, self.slots)

    def latent(self, context, frame, budget):
        return predict(self.backbone, self.model, frame.read(budget), frame.fit, frame.space, self.sigma, context)

    @torch.no_grad()
    def warmup(self, state, image, frame_id):
        sample = capture(self.backbone, state, frame_id, image)
        budget = REF_BUDGET[sample.context.intra] if self.run.sparse else DENSE
        aligned = self.latent(sample.context, self.receive(sample), budget)
        return commit_reference(sample.context, aligned, self.projector(aligned, sample.context.intra))

    def rgb_objective(self, aligned, sample, frame, crops):
        # Reference latents and teacher are FP32. Autocast only the frozen
        # reconstruction network while checkpointing its input gradient.
        context = sample.context
        if context.context is not None and len(aligned) > 1:
            context = replace(context, context=context.context.expand(len(aligned), -1, -1, -1))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            reconstructed, _ = self.backbone.decode_aligned(aligned, context, checkpointed=True)
        target_rgb = sample.rgb.expand_as(reconstructed)
        loss, _ = self.objective(target_rgb, reconstructed.float(), crops)
        anchor = weighted_anchor(aligned, sample.aligned, frame.fit, self.backbone, frame.space)
        return loss + ANCHOR_WEIGHT * anchor

    def frame_gradients(self, sample, frame, budgets, crops, *, update):
        context = sample.context
        latents = [self.latent(context, frame, k) for k in budgets]
        reference = latents[0]
        aligned = reference if len(latents) == 1 else torch.cat(latents)
        if update and not aligned.requires_grad:
            raise RuntimeError("recovery latent lost its gradient at the frozen Decoder")
        loss = self.rgb_objective(aligned, sample, frame, crops)
        ref, projected = reference_supervision(self.backbone, self.projector, reference, sample.native, context)
        loss = loss + REFERENCE_WEIGHT * ref
        if update:
            (loss / self.run.length).backward()
        return loss.detach(), (reference.detach(), projected.detach())

    def sequence(self, clip, phase, step_seed, *, update, verify_sync=False, check_identity=False):
        state, replica = ReferenceState(), ReferenceState()
        rng = np.random.default_rng(step_seed)
        crop_rng = torch.Generator().manual_seed(step_seed)
        total, diagnostic = 0.0, []
        for frame_id, image in enumerate(clip):
            if frame_id < phase:
                state = self.warmup(state, image, frame_id)
                if verify_sync:
                    replica = self.warmup(replica, image, frame_id)
                continue
            sample = capture(self.backbone, state, frame_id, image)
            frame = self.receive(sample)
            budgets = budgets_for(sample.context, self.run.sparse, rng)
            if check_identity:
                with torch.no_grad():
                    base = sparse_latent(self.backbone, frame.fit, frame.space, frame.read(budgets[0]), self.sigma)
                    if not torch.equal(base, self.latent(sample.context, frame, budgets[0])):
                        raise RuntimeError("step 0 recovery did not reproduce its frozen baseline")
            crops = crop_plan(*sample.rgb.shape[-2:], crop_rng)
            loss, (aligned, feature) = self.frame_gradients(sample, frame, budgets, crops, update=update)
            total += loss
            state = commit_reference(sample.context, aligned, feature)
            if verify_sync:
                with torch.no_grad():
                    rx = self.backbone.begin_frame(replica, frame_id)
                    recovered = self.latent(rx, frame, budgets[0])
                    replica = commit_reference(rx, recovered, self.projector(recovered, rx.intra))
                if not same_state(state, replica):
                    raise RuntimeError(f"C1 training reference desynchronized at {frame_id}")
                diagnostic.append({"frame": frame_id, "frame_type": sample.frame_type, "budgets": budgets,
                                   "reference_budget": budgets[0], "state_sync": True})
        return total.item() / self.run.length, diagnostic

    def step(self, clip, phase, step_seed, *, verify_sync=False):
        self.optimizer.zero_grad(set_to_none=True)
        loss, rows = self.sequence(clip, phase, step_seed, update=True, verify_sync=verify_sync)
        norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True))
        if not math.isfinite(loss) or norm <= 0:
            raise RuntimeError(f"invalid C1 update loss={loss}, gradient norm={norm}")
        self.optimizer.step()
        return {"loss": loss, "gradient_norm": norm, "frames": rows}


def memory():
    return round(torch.cuda.max_memory_allocated() / 2**20, 2)


def smoke(train, codec, fits, lock, run, loss_weights, *, phase=0):
    trainer = Trainer(codec, fits, lock, run, loss_weights)
    clip = train.read("DAVIS2017:bear", 0, run.length + phase)
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    with torch.no_grad():
        initial, _ = trainer.sequence(clip, phase, run.seed, update=False, check_identity=True)
    updates = []
    for index in range(SMOKE_UPDATES):
        tick = time.perf_counter()
        update = trainer.step(clip, phase, run.seed, verify_sync=True)
        updates.append({"step": index + 1, "elapsed_s": time.perf_counter() - tick, **update})
    with torch.no_grad():
        final, _ = trainer.sequence(clip, phase, run.seed, update=False, verify_sync=True)
    passed = math.isfinite(final) and final < initial
    result = {"run": asdict(run), "rollout": run.rollout, "start_phase": phase, "step_0_identity": True,
              "scan_lock_sha256": file_hash(LOCKED), "initial_loss": initial, "final_loss": final, "steps": updates,
              "passed": passed, "seconds": time.perf_counter() - start, "peak_allocated_mib": memory(),
              "median_seconds_per_update": statistics.median(r["elapsed_s"] for r in updates)}
    write_json(OUTPUT / run.phase / f"smoke_{run.length}f.json", result)
    print(f"{run.phase} {run.rollout} smoke: {initial:.5f} -> {final:.5f}; "
          f"{result['median_seconds_per_update']:.3f}s/update, {result['peak_allocated_mib']:.0f} MiB", flush=True)
    return result


def seed_path(run):
    return WEIGHTS / run.phase / f"seed{run.seed}.pt"


def plan_identity(run, fits_identity, loss_weights):
    return {
        "run": asdict(run), "backbone_identity": fits_identity, "scan_lock_sha256": file_hash(LOCKED),
        "objective": {"reconstruction": loss_weights, "latent_anchor": ANCHOR_WEIGHT, "reference": REFERENCE_WEIGHT},
        "lr": LR, "fp32_reference": True, "bf16_checkpointed_decoder": True, "rollout": run.rollout,
    }


def load_model(run, identity):
    artifact = torch.load(seed_path(run), weights_only=True, map_location="cuda:0")
    if artifact["identity"] != identity or artifact["step"] != run.steps:
        raise RuntimeError("C1 closed-loop evaluation requires a complete matching training seed")
    model = RecoveryNet().cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(artifact["model"])
    return model.eval().requires_grad_(False)


def resume_training(trainer, plan, identity, path):
    if not path.is_file():
        return 0, []
    saved = torch.load(path, weights_only=True, map_location="cuda:0")
    if saved["identity"] != identity or saved["plan"] != plan:
        raise RuntimeError(f"checkpoint identity differs for {path}")
    trainer.model.load_state_dict(saved["model"])
    trainer.optimizer.load_state_dict(saved["optimizer"])
    return saved["step"], saved["history"]


def save_training(trainer, plan, history, run, identity):
    step = len(history)
    path = seed_path(run)
    save_checkpoint(path, {"identity": identity, "plan": plan, "step": step, "history": history,
                           "model": trainer.model.state_dict(), "optimizer": trainer.optimizer.state_dict()})
    path_out = OUTPUT / run.phase / f"seed{run.seed}_training.json"
    elapsed = sum(row["elapsed_s"] for row in history)
    write_json(path_out, {"identity": identity, "steps": step, "target": run.steps, "plan": plan, "updates": history,
                          "checkpoint_sha256": file_hash(path), "seconds": elapsed,
                          "updates_per_second": step / elapsed, "peak_allocated_mib": max(
                              row["peak_allocated_mib"] for row in history),
                          "videos": len({row["source"] for row in plan[:step]}),
                          "start_phases": sorted({row["phase"] for row in plan[:step]})})
    print(f"{run.phase} seed={run.seed} step={step}/{run.steps} loss={history[-1]['loss']:.5f} "
          f"{elapsed / step:.3f}s/update", flush=True)


def train_seed(codec, fits, lock, run, loss_weights, catalog, source_identity):
    plans = make_plans(catalog, run)
    identity = plan_identity(run, source_identity, loss_weights)
    trainer = Trainer(codec, fits, lock, run, loss_weights)
    start, history = resume_training(trainer, plans, identity, seed_path(run))
    subset = plans[start:]
    rows = [catalog.rows[k] for k in sorted(catalog.rows)]
    torch.cuda.reset_peak_memory_stats()
    for index, (clip, plan) in enumerate(zip(prefetched_windows(rows, subset), subset), start + 1):
        tick = time.perf_counter()
        step_seed = 100_000 * run.seed + index
        record = trainer.step(clip, plan["phase"], step_seed)
        history.append({"step": index, "loss": record["loss"], "gradient_norm": record["gradient_norm"],
                        "elapsed_s": time.perf_counter() - tick, "peak_allocated_mib": memory()})
        if index % SAVE_EVERY == 0 or index == run.steps:
            save_training(trainer, plans, history, run, identity)
    if len(history) != run.steps:
        raise RuntimeError("C1 training ended before its declared update count")
    return identity
