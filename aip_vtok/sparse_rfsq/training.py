"""One-seed W1/W2 pilot with separate causal references and identical schedules."""

import math
import time
from dataclasses import replace
from itertools import combinations

import numpy as np
import torch
from torch.nn import functional as F

from aip_vtok.c1.artifacts import timestamp
from aip_vtok.c1.frames import capture
from aip_vtok.c1.recovery_net import reference_supervision
from aip_vtok.c1.scan import BUDGETS, REF_BUDGET
from aip_vtok.c1.training import LR, Run, make_plans, memory, weighted_anchor
from aip_vtok.c1_followup.io import create_record, tensor_digest, write_rows
from aip_vtok.diagnostics.diag0.io import json_at
from aip_vtok.evaluation.fix import HORIZON_VIDEO
from aip_vtok.runtime import file_hash, save_checkpoint, write_json
from aip_vtok.training.losses import ReconstructionLoss, crop_plan

from .calibration import training_catalog
from .evaluation import CLOSED, FIXED, S1, S2, Trial, evaluate, load_s1, summaries
from .gates import pilot_decision
from .protocol import (
    ANCHOR_WEIGHT,
    DIAG_EVERY,
    OUTPUT,
    PILOTS,
    REFERENCE_WEIGHT,
    S1_GATE,
    S2_GATE,
    SCAN_LOCK,
    SEED,
    SMOKE_LENGTHS,
    UPDATES,
    WEIGHTS,
)
from .receiver import B2, B3, Receiver, RecoveryNet
from .rollout import CausalPair

TERMS = ("mse", "lpips", "anchor")


def gradient_vectors(terms, parameters):
    return {name: torch.cat([grad.detach().float().flatten() for grad in torch.autograd.grad(
        terms[name], parameters, retain_graph=True
    )]) for name in TERMS}


def summarize_gradients(vectors):
    norms = {name: torch.linalg.vector_norm(vector).item() for name, vector in vectors.items()}
    if not all(math.isfinite(norm) for norm in norms.values()):
        raise RuntimeError("nonfinite recovery trunk gradient")
    cosines = {f"{a}/{b}": F.cosine_similarity(vectors[a], vectors[b], dim=0, eps=1e-30).clamp(-1, 1).item()
               if norms[a] and norms[b] else None for a, b in combinations(vectors, 2)}
    return {"norms": norms, "cosines": cosines}


def budgets_for(intra, rng):
    ref = REF_BUDGET[intra]
    middle = int(rng.choice([k for k in BUDGETS[1:-1] if k != ref]))
    return ref, middle, BUDGETS[-1]


def create_model():
    torch.manual_seed(SEED)
    return RecoveryNet().cuda().to(memory_format=torch.channels_last)


class Trainer:
    def __init__(self, run, cases, lock, lpips):
        self.codec, self.fits, self.cases = run.codec, run.inputs.fits, cases
        self.model = create_model()
        self.receiver = Receiver(lock["sigma"], self.model)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=LR, fused=True)
        self.loss_weights = run.config["stage1"]["loss"] | {"lpips": lpips}
        self.objective = ReconstructionLoss(self.loss_weights).cuda().to(memory_format=torch.channels_last)

    def objective_terms(self, sample, case, latents, crops):
        aligned = torch.cat(latents)
        if not aligned.requires_grad:
            raise RuntimeError("frozen reconstruction decoder lost the recovery input gradient")
        ctx = sample.context
        if not ctx.intra:
            ctx = replace(ctx, context=ctx.context.expand(len(latents), -1, -1, -1))
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            rec, _ = self.codec.backbone.decode_aligned(aligned, ctx, checkpointed=True)
        rgb_loss, parts = self.objective(sample.rgb.expand_as(rec), rec.float(), crops)
        anchor = weighted_anchor(aligned, sample.aligned, self.fits[sample.frame_type], self.codec.backbone,
                                 case.layout.space)
        ref_loss, feature = reference_supervision(self.codec.backbone, self.codec.projector, latents[0],
                                                   sample.native, sample.context)
        return rgb_loss + ANCHOR_WEIGHT * anchor + REFERENCE_WEIGHT * ref_loss, parts | {"anchor": anchor}, ref_loss, feature

    def frame_update(self, pair, image, index, rng, crop_rng, length, *, diagnose, smoke):
        backbone = self.codec.backbone
        sample = capture(backbone, pair.tx, index, image)
        case = self.cases[sample.frame_type]
        payload = case.encode(sample)
        budgets = budgets_for(sample.context.intra, rng)
        received = [payload[:, :k] for k in budgets]
        latents = [self.receiver(case, backbone, sample.context, part) for part in received]
        crops = crop_plan(*sample.rgb.shape[-2:], crop_rng)
        loss, terms, ref_loss, feature = self.objective_terms(sample, case, latents, crops)
        reference_norm = self.check_step_zero(case, sample.context, received, latents, ref_loss) if smoke else None
        vectors = gradient_vectors(terms, tuple(self.model.body.parameters())) if diagnose else None
        (loss / length).backward()
        with torch.no_grad():
            ctx = backbone.begin_frame(pair.rx, index)
            reconstructed = self.receiver(case, backbone, ctx, received[0])
            projected = self.codec.projector(reconstructed, ctx.intra)
            pair.commit(sample.context, latents[0], feature, ctx, reconstructed, projected)
        return loss.detach(), vectors, {"frame": index, "kind": sample.frame_type, "budgets": budgets,
                                        "reference_gradient_norm": reference_norm, "state_sync": True}

    def check_step_zero(self, case, context, received, latents, ref_loss):
        with torch.no_grad():
            base = Receiver(self.receiver.sigma)
            for part, latent in zip(received, latents):
                if not torch.equal(latent, base(case, self.codec.backbone, context, part)):
                    raise RuntimeError("zero-initialized B3 differs from the frozen B2 baseline")
        gradient, = torch.autograd.grad(ref_loss, latents[0], retain_graph=True)
        norm = gradient.norm().item()
        if not math.isfinite(norm) or norm <= 0:
            raise RuntimeError("reference supervision has no finite gradient through the projector")
        return norm

    def step(self, clip, phase, step, *, diagnose=False, smoke=False):
        if not phase < len(clip):
            raise ValueError("a training window must contain its supervised suffix")
        self.optimizer.zero_grad(set_to_none=True)
        rng_seed = 100_000 * SEED + step
        rng, crop_rng = np.random.default_rng(rng_seed), torch.Generator().manual_seed(rng_seed)
        pair = CausalPair()
        for index in range(phase):
            pair.advance(self.codec, clip[index], index, self.cases, self.receiver)
        length = len(clip) - phase
        total, rows = 0.0, []
        accum = dict.fromkeys(TERMS, 0)
        for index in range(phase, len(clip)):
            loss, vectors, row = self.frame_update(pair, clip[index], index, rng, crop_rng, length,
                                                    diagnose=diagnose, smoke=smoke)
            rows.append(row)
            total += loss
            if vectors is not None:
                for name, vector in vectors.items():
                    accum[name] = accum[name] + vector / length
        gradient_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True).item()
        loss = total.item() / length
        if not math.isfinite(loss) or gradient_norm <= 0:
            raise RuntimeError("sparse recovery got a nonfinite loss or zero gradient")
        self.optimizer.step()
        return {"step": step, "rng_seed": rng_seed, "phase": phase, "frames": rows, "loss": loss,
                "gradient_norm": gradient_norm, "gradient_diagnostics": summarize_gradients(accum) if diagnose else None}


def smoke(run, train, lock, cases, plans):
    source = sorted(train.rows)[plans[0]["source"]]
    for length in SMOKE_LENGTHS:
        path = OUTPUT / f"smoke_{length}f.json"
        if path.is_file():
            saved = json_at(path)
            if not saved["passed"] or saved["scan_lock_sha256"] != file_hash(SCAN_LOCK):
                raise RuntimeError(f"stale {length}-frame smoke record")
            continue
        trainer = Trainer(run, cases, lock, PILOTS["W1"])
        phase = 0 if length == 2 else 12
        clip = run.reader.read(train, source, 0, phase + length)
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        result = trainer.step(clip, phase, 1, smoke=True)
        record = {**result, "video": source, "length": length, "elapsed_s": time.perf_counter() - start,
                  "peak_allocated_mib": memory(), "step_zero_b2_identity": True, "passed": True,
                  "scan_lock_sha256": file_hash(SCAN_LOCK), "integrity": run.verify_integrity()}
        write_json(path, record)
        print(f"S2 {length}-frame smoke: loss={result['loss']:.6f}, GPU peak={record['peak_allocated_mib']:.0f} MiB",
              flush=True)


def checkpoint_path(variant):
    return WEIGHTS / variant / f"seed{SEED}.pt"


def pilot_identity(trainer, variant):
    return {"variant": variant, "scan_lock_sha256": file_hash(SCAN_LOCK), "seed": SEED, "updates": UPDATES,
            "rollout_frames": SMOKE_LENGTHS[-1], "init_sha256": tensor_digest(trainer.model),
            "loss": trainer.loss_weights | {"latent_anchor": ANCHOR_WEIGHT, "reference": REFERENCE_WEIGHT}, "lr": LR}


def resume(trainer, path, identity, plans):
    if not path.is_file():
        return []
    data = torch.load(path, weights_only=True, map_location="cuda:0")
    if data["identity"] != identity or data["plan"] != plans or data["step"] != len(data["history"]):
        raise RuntimeError("sparse pilot checkpoint disagrees with the locked data, training settings or steps")
    if data["step"] > UPDATES or any(row["step"] != i for i, row in enumerate(data["history"], 1)):
        raise RuntimeError("sparse pilot checkpoint has an invalid update history")
    trainer.model.load_state_dict(data["model"])
    trainer.optimizer.load_state_dict(data["optimizer"])
    return data["history"]


def save_progress(trainer, path, identity, plans, history):
    step = len(history)
    save_checkpoint(path, {"identity": identity, "step": step, "plan": plans, "history": history,
                           "model": trainer.model.state_dict(), "optimizer": trainer.optimizer.state_dict()})
    elapsed = sum(row["elapsed_s"] for row in history)
    print(f"S2 {identity['variant']} {step}/{UPDATES}, loss={history[-1]['loss']:.6f}, {elapsed / step:.3f}s/update",
          flush=True)


def train_variant(run, train, cases, lock, plans, variant, lpips):
    trainer = Trainer(run, cases, lock, lpips)
    path, identity = checkpoint_path(variant), pilot_identity(trainer, variant)
    history = resume(trainer, path, identity, plans)
    pending = plans[len(history):]
    torch.cuda.reset_peak_memory_stats()
    for clip, window in zip(run.reader.training_windows(train, pending), pending):
        step = len(history) + 1
        start = time.perf_counter()
        update = trainer.step(clip, window["phase"], step, diagnose=step % DIAG_EVERY == 0)
        history.append(update | {"elapsed_s": time.perf_counter() - start, "peak_allocated_mib": memory()})
        if step % DIAG_EVERY == 0 or step == UPDATES:
            save_progress(trainer, path, identity, plans, history)
    expected = list(range(DIAG_EVERY, UPDATES + 1, DIAG_EVERY))
    if len(history) != UPDATES or [r["step"] for r in history if r["gradient_diagnostics"] is not None] != expected:
        raise RuntimeError("sparse pilot did not finish all updates and required diagnostics")
    record = {"identity": identity, "steps": len(history), "updates": history, "checkpoint_sha256": file_hash(path),
              "seconds": sum(row["elapsed_s"] for row in history), "integrity": run.verify_integrity()}
    write_json(OUTPUT / f"{variant}_training.json", record)
    return trainer.model.eval().requires_grad_(False), record


def compare_training(records):
    w1, w2 = (records[name] for name in PILOTS)
    if w1["identity"]["init_sha256"] != w2["identity"]["init_sha256"]:
        raise RuntimeError("W1/W2 initialization differs")
    keys = ("step", "rng_seed", "phase", "frames")
    if [[r[k] for k in keys] for r in w1["updates"]] != [[r[k] for k in keys] for r in w2["updates"]]:
        raise RuntimeError("W1/W2 consumed different budget, crop RNG or causal frame schedules")


def baseline_rows(rows, winner):
    return [r for r in rows if r["strategy"] == winner and r["method"] == B2]


def verify_fixed_history(base, learned):
    fields = ("reference_latent_rms", "projected_feature_rms", "rx_reference_sha256", "prefix_ids_match")
    reference = {(row["video"], row["frame"], row["budget"]): tuple(row.get(k) for k in fields)
                 for row in base if row["history"] == FIXED}
    if any(reference[row["video"], row["frame"], row["budget"]] != tuple(row.get(k) for k in fields)
           for row in learned if row["history"] == f"{FIXED}_B3"):
        raise RuntimeError("fixed_reference_B3 disagrees with the S1 B2 anchor")


def score_pilot(run, lock, cases, models, results):
    winner = lock["winner"]
    trials = [Trial(winner, B3, cases, Receiver(lock["sigma"], model), label) for label, model in models.items()]
    rows, counts = evaluate(run, lock, trials)
    s1 = json_at(S1.with_suffix(".json"))
    if counts != s1["cohort"]:
        raise RuntimeError("S2 Validation does not match the locked S1 cohort")
    base = baseline_rows(s1["rows"], winner)
    verify_fixed_history(base, rows)
    base_summary, summary = summaries(base), summaries(rows)
    integrity = run.verify_integrity()
    decisions = {}
    for label in results:
        rms = max(r["reference_latent_rms"] for r in rows if r["variant"] == label
                  and r["history"] == f"{CLOSED}_B3" and r["video"] == HORIZON_VIDEO)
        decisions[label] = pilot_decision(base_summary, summary, label, rms, integrity["passed"])
    write_rows(S2, rows, summaries=summary, cohort=counts, scan_lock_sha256=file_hash(SCAN_LOCK))
    return decisions, integrity


def run_pilot(run, lock, tests):
    s1 = load_s1(lock)
    if not s1["released"]:
        create_record(S2_GATE, {"started": False, "recommend_s3": False, "reason": s1["reason"],
                                "s1_gate_sha256": file_hash(S1_GATE)})
        print(f"S2 withheld: {s1['reason']}", flush=True)
        return
    if S2_GATE.is_file():
        saved = json_at(S2_GATE)
        if saved["s1_gate_sha256"] != file_hash(S1_GATE) or saved["s2_sha256"] != file_hash(S2.with_suffix(".json")):
            raise RuntimeError("published S2 gate disagrees with its evidence")
        print(f"S2 already complete. Recommend S3: {saved['recommend_s3']}", flush=True)
        return
    train = training_catalog(run.reader.open("train"), lock["holdout"])
    cases = run.inputs.cases(lock["winner"], lock["keys"][lock["winner"]])
    plans = make_plans(train, Run("sparse-rfsq", SEED, UPDATES, SMOKE_LENGTHS[-1], late_gop=True))
    create_record(OUTPUT / "training_plan.json", {"videos": sorted(train.rows), "windows": plans,
                                                 "scan_lock_sha256": file_hash(SCAN_LOCK)})
    before_validation = run.reader.counts["validation"]
    smoke(run, train, lock, cases, plans)
    models, records = {}, {}
    for label, lpips in PILOTS.items():
        models[label], records[label] = train_variant(run, train, cases, lock, plans, label, lpips)
    compare_training(records)
    if run.reader.counts["validation"] != before_validation:
        raise RuntimeError("Validation images are unavailable during S2 training")
    decisions, integrity = score_pilot(run, lock, cases, models, records)
    record = {"started": True, "finished": timestamp(), "winner": lock["winner"], "decisions": decisions,
              "s1_gate_sha256": file_hash(S1_GATE), "s2_sha256": file_hash(S2.with_suffix(".json")), "tests": tests,
              "weights": {name: r["checkpoint_sha256"] for name, r in records.items()}, "integrity": integrity,
              "recommend_s3": any(d["passed"] for d in decisions.values())}
    create_record(S2_GATE, record)
    print(f"S2 gate: {[(name, d['checks']) for name, d in decisions.items()]}", flush=True)
