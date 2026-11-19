"""Conditional C1-B-Followup validation of sparse integer prefixes."""

import math
from bisect import bisect_right
from dataclasses import dataclass
from itertools import islice

import torch

from aip_vtok.c1.artifacts import timestamp
from aip_vtok.c1.frames import capture
from aip_vtok.c1.metrics import FrameMetrics
from aip_vtok.c1.recovery_net import commit_reference
from aip_vtok.c1.scan import BUDGETS, REF_BUDGET, SHAPE, gaussian_fill, halton_positions
from aip_vtok.c1.tokens import FRAME_HEADER, PACKET_TOKENS
from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.diagnostics.diag0.closed import same_state
from aip_vtok.diagnostics.diag0.io import json_at
from aip_vtok.runtime import file_hash, write_json

from .dense import summarize_arms
from .experiment import PHASE1, frozen_candidates, load_layouts, load_phase1, validation_input
from .gates import LIMITS, sparse_release
from .io import CANDIDATES, OUTPUT, create_record, write_rows
from .quantization import CODE_BITS, LEVEL_HEADER, wire_ledger
from .rollout import scored_row
from .training import TRAINING_GATE, SeedRun, read_milestone, read_weights, recovery_model

SPARSE = OUTPUT / "c1b_followup"
SPARSE_GATE = OUTPUT / "sparse_gate.json"
SIGMA = 1.5


def scan_slots(shape, weights, length=None):
    """Static S3 quotas over deduplicated Halton positions, keyed by (group, y, x)."""
    height, width = shape
    area = height * width
    capacity = area * len(weights)
    length = capacity if length is None else length
    if not weights or min(weights) <= 0 or not 0 <= length <= capacity or min(shape) <= 0:
        raise ValueError("invalid sparse scan geometry, weights or length")
    positions = halton_positions(height, width)
    used = [0] * len(weights)
    total = sum(weights)
    for tick in range(1, length + 1):
        available = (group for group in range(len(weights)) if used[group] < area)
        group = max(available, key=lambda g: tick * weights[g] - total * used[g])
        yield group, *divmod(positions[used[group]], width)
        used[group] += 1


def progressive_slots(slots, stages):
    """Send each coarse chunk before its refinement chunk. Prefixes may truncate either."""
    if stages not in (1, 2):
        raise ValueError("sparse FSQ requires one or two levels")
    iterator = iter(slots)
    while chunk := tuple(islice(iterator, PACKET_TOKENS)):
        for level in range(1, stages + 1):
            yield from ((level, group, y, x) for group, y, x in chunk)


@dataclass(frozen=True)
class SparsePlan:
    schedule: torch.Tensor
    shape: tuple
    groups: int

    @classmethod
    def create(cls, schedule, shape, groups, device):
        keys = torch.tensor(schedule, device=device, dtype=torch.long).reshape(-1, 4)
        height, width = shape
        levels, group, y, x = keys.unbind(1)
        if not ((levels >= 1) & (levels <= 2) & (group >= 0) & (group < groups)
                & (y >= 0) & (y < height) & (x >= 0) & (x < width)).all():
            raise ValueError("sparse slot outside the frozen layout")
        flat = (group * height + y) * width + x
        if len(torch.unique((levels - 1) * groups * height * width + flat)) != len(flat):
            raise ValueError("duplicate (group, y, x) within a sparse level")
        coarse = torch.full((groups * height * width,), len(flat), device=device, dtype=torch.long)
        primary = levels == 1
        coarse[flat[primary]] = torch.arange(len(flat), device=device)[primary]
        if (coarse[flat[~primary]] >= torch.arange(len(flat), device=device)[~primary]).any():
            raise ValueError("Level 2 requires the corresponding earlier Level 1 slot")
        return cls(torch.stack((levels - 1, flat)), shape, groups)

    def encode_prefix(self, ids, count):
        level, slot = self.prefix(count)
        return torch.stack(ids, dim=1).flatten(2)[:, level, slot]

    def prefix(self, count):
        if not 0 <= count <= self.schedule.shape[1]:
            raise ValueError("sparse prefix exceeds the frozen schedule")
        return self.schedule[:, :count]

    def values_and_mask(self, quantizer, payload):
        if payload.ndim != 2 or payload.dtype != torch.int64:
            raise ValueError("sparse packets require [B, N] integer IDs")
        level, slot = self.prefix(payload.shape[1])
        stages = len(quantizer.stages)
        if (level >= stages).any():
            raise ValueError("received Level 2 for single-stage FSQ")
        area = self.groups * math.prod(self.shape)
        offset = level * area + slot
        canvas = payload.new_zeros((len(payload), stages * area))
        canvas[:, offset] = payload
        mask = torch.zeros_like(canvas, dtype=torch.bool)
        mask[:, offset] = True
        shape = (len(payload), stages, self.groups, *self.shape)
        return sparse_values(quantizer, canvas.reshape(shape), mask.reshape(shape))


def sparse_values(quantizer, ids, masks):
    primary = masks[:, 0]
    reconstructed = quantizer.primary.decode(ids[:, 0])
    if quantizer.residual is not None:
        refinement = masks[:, 1]
        if (refinement & ~primary).any():
            raise ValueError("Level 2 requires the corresponding Level 1")
        delta = quantizer.residual.decode(ids[:, 1])
        reconstructed = reconstructed + torch.where(refinement[..., None], delta, 0)
    values = torch.where(primary[..., None], reconstructed, 0)
    return values, primary


def interpolate(values, mask, sigma=SIGMA):
    return gaussian_fill(values.movedim(-1, 2), mask, sigma).movedim(2, -1)


def budget_prefixes(schedule, entropy):
    """Select full legal prefixes under common absolute caps, using Train entropy."""
    counts = [0, 0]
    frame_bits = FRAME_HEADER.size * 8
    raw, estimated, slots = [frame_bits], [frame_bits], [0]
    for level, group, _, _ in schedule:
        stage = level - 1
        overhead = LEVEL_HEADER.size * 8 if counts[stage] % PACKET_TOKENS == 0 else 0
        counts[stage] += 1
        raw.append(raw[-1] + overhead + CODE_BITS)
        estimated.append(estimated[-1] + overhead + entropy[stage][group])
        slots.append(counts[0])
    for limit in BUDGETS:
        bit_cap = wire_ledger([limit], CODE_BITS)["raw_total_bytes"] * 8
        for kind, trace, cap in (("fixed_slots", slots, limit), ("fixed_bits", raw, bit_cap),
                                 ("train_entropy_budget", estimated, bit_cap)):
            n = bisect_right(trace, cap) - 1
            yield {"budget": limit, "budget_type": kind, "cap": cap, "prefix_symbols": n,
                   "unique_slots": slots[n], "level2_symbols": n - slots[n], "raw_bits": raw[n],
                   "train_entropy_est_bits": estimated[n], "real_bitstream": False}


def sparse_case(arm, kind, layout, calibration, stages):
    energy = [calibration["sensitivity"][kind][i] for i in layout.indices.tolist()]
    per_group = [sum(energy[i:i + 6]) for i in range(0, len(energy), 6)]
    total = max(sum(per_group), 1e-30)
    weights = [max(1, round(value * 1_000_000 / total)) for value in per_group]
    slots = tuple(scan_slots(SHAPE, weights))
    entropy = [[group["id_entropy_bits"] for group in record["groups"]]
               for record in calibration["histograms"] if record["arm"] == arm and record["frame_type"] == kind]
    if len(entropy) != stages or any(len(level) != layout.groups for level in entropy):
        raise RuntimeError("sparse budgets require the frozen Train entropy of each group and level")
    budgets = list(budget_prefixes(progressive_slots(slots, stages), entropy))
    return {"weights": weights, "slots": list(map(list, slots)), "entropy": entropy, "budgets": budgets}


def lock_sparse(arm, layouts, calibration, seed_records):
    stages = len(layouts["I"].quantizer.stages)
    cases = {kind: sparse_case(arm, kind, layout, calibration, stages) for kind, layout in layouts.items()}
    path = SPARSE / arm / "scan_lock.json"
    create_record(path, {"arm": arm, "case": cases, "candidate_lock_sha256": file_hash(CANDIDATES),
                         "checkpoint_sha256": [row["checkpoint_sha256"] for row in seed_records],
                         "sigma": SIGMA, "shape": list(SHAPE), "stages": stages})
    return cases


def prefix_base(layout, plan, ids, n, backbone):
    values, mask = plan.values_and_mask(layout.quantizer, plan.encode_prefix(ids, n))
    return layout.restore(interpolate(values, mask), backbone), mask


def sparse_scores(run, sample, context, video, layout, plan, ids, budgets, models, sync):
    scorer = FrameMetrics(run.suite, sample.rgb)
    rows = {}
    for budget in budgets:
        n = budget["prefix_symbols"]
        if n not in rows:
            base, mask = prefix_base(layout, plan, ids, n, run.codec.backbone)
            outputs = {"B0-Gaussian": base}
            outputs.update({method: model(base, mask, layout.indices, context) for method, model in models.items()})
            rows[n] = [scored_row(run.codec.backbone, sample, context, latent, sync, scorer,
                                  video, "same_sparse_gaussian_history", method)
                       for method, latent in outputs.items()]
        yield from (row | budget | {"arm": row["method"]} for row in rows[n])


@torch.no_grad()
def sparse_video(run, video, clip, layouts, cases, plans, models):
    tx, rx = ReferenceState(), ReferenceState()
    backbone, projector = run.codec.backbone, run.codec.projector
    for frame_id, image in enumerate(clip):
        sample = capture(backbone, tx, frame_id, image)
        context = backbone.begin_frame(rx, frame_id)
        kind = sample.frame_type
        layout, plan = layouts[kind], plans[kind]
        ids, _ = layout.encode(sample)
        budgets = cases[kind]["budgets"]
        reference = next(b for b in budgets if b["budget_type"] == "fixed_slots" and b["budget"] == REF_BUDGET[context.intra])
        ref_n = reference["prefix_symbols"]
        tx_latent, _ = prefix_base(layout, plan, ids, ref_n, backbone)
        rx_latent, _ = prefix_base(layout, plan, ids, ref_n, backbone)
        subprefix = plan.encode_prefix(ids, ref_n // 2)
        if not torch.equal(subprefix, plan.encode_prefix(ids, ref_n)[:, :ref_n // 2]):
            raise RuntimeError("sparse sub-prefix changed its integer IDs")
        tx = commit_reference(sample.context, tx_latent, projector(tx_latent, context.intra))
        rx = commit_reference(context, rx_latent, projector(rx_latent, context.intra))
        sync = same_state(tx, rx)
        if not sync:
            raise RuntimeError(f"{video} sparse reference desynchronized at frame {frame_id}")
        yield from sparse_scores(run, sample, context, video, layout, plan, ids, budgets, models, sync)


def release_seeds(arm, outcome, integrity):
    targets = ("seed1_final", "seed2_final")
    declared = [outcome.get(key) or {} for key in targets]
    if not sparse_release(True, *declared, integrity, LIMITS):
        raise RuntimeError("C1-B-Followup requires two passing seeds and intact sealed inputs")
    if not outcome["pilot"]["gate"]["passed"]:
        raise RuntimeError("C1-B-Followup requires the earlier 500-step Pilot")
    jobs = [SeedRun(arm, entry["seed"], entry["identity"], []) for entry in declared]
    records = [read_milestone(job, LIMITS["final_steps"]) for job in jobs]
    if records != declared:
        raise RuntimeError("C1-B-Followup records differ from the evaluated milestones")
    return jobs, records


def run_sparse_arm(run, arm, layouts, calibration, jobs, records):
    cases = lock_sparse(arm, layouts, calibration, records)
    groups = layouts["I"].groups
    stages = len(layouts["I"].quantizer.stages)
    plans = {kind: SparsePlan.create(tuple(progressive_slots(cases[kind]["slots"], stages)),
                                     SHAPE, groups, "cuda") for kind in cases}
    models = {}
    for job in jobs:
        model = recovery_model(groups, "cuda")
        saved = read_weights(job.checkpoint(LIMITS["final_steps"]), job.identity)
        model.load_state_dict(saved["model"])
        models[f"B1-seed{job.seed}"] = model.eval().requires_grad_(False)
    catalog, counts = validation_input(run)
    rows = []
    for video, count in counts.items():
        clip = run.reader.read(catalog, video, 0, count)
        rows.extend(sparse_video(run, video, clip, layouts, cases, plans, models))
        print(f"c1b_followup {arm}, {video}, {count} frames per point", flush=True)
    by_point = {}
    for row in rows:
        point = (row["budget_type"], row["budget"])
        by_point.setdefault(point, []).append(row)
    summaries = [{"budget_type": kind, "budget": budget, "metrics": summarize_arms(chunk)}
                 for (kind, budget), chunk in by_point.items()]
    write_rows(SPARSE / arm / "validation", rows, summaries=summaries, cohort=counts, state_sync=True)
    return {"arm": arm, "frames": sum(counts.values()), "points": len(summaries), "state_sync": True}


def run_sparse(run):
    phase1 = load_phase1()
    training = json_at(TRAINING_GATE)
    if training["phase1_sha256"] != file_hash(PHASE1):
        raise RuntimeError("sparse gate input changed since dense calibration")
    integrity = run.verify_integrity()
    released = training["sparse_released"]
    if any(arm not in phase1["released"] for arm in released):
        raise RuntimeError("sparse validation requested for a candidate withheld by Phase 1")
    done = []
    if released:
        layouts = load_layouts(frozen_candidates(), "cuda")
        calibration = json_at(OUTPUT / "calibration.json")
        for arm in released:
            jobs, records = release_seeds(arm, training["candidates"][arm], integrity)
            done.append(run_sparse_arm(run, arm, layouts[arm], calibration, jobs, records))
    elif SPARSE.exists():
        raise RuntimeError("sparse scan assets exist without a passing dual-seed gate")
    write_json(SPARSE_GATE, {"label": "c1b_followup", "started_validation": bool(done), "released": released,
                            "completed": done, "training_steps": 0, "finished": timestamp(),
                            "training_gate_sha256": file_hash(TRAINING_GATE), "integrity": run.verify_integrity()})
    print(f"c1b_followup started: {bool(done)}, released arms: {released}", flush=True)
