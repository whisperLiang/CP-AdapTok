"""Derive every new quantizer and the shortlist from held-out Train windows."""

from collections import defaultdict
from itertools import groupby, product

import torch

from aip_vtok.c1.calibrate import channel_rows, key, load_fits, sample_gradient
from aip_vtok.c1.channels import ALIGNED, CHANNELS, coordinate_value
from aip_vtok.c1.training import Run, make_plans
from aip_vtok.diagnostics.diag0.io import json_at
from aip_vtok.diagnostics.diag0.sparse import fit_scale
from aip_vtok.runtime import file_hash

from .dense import Histograms, Layout, evaluate_samples, reconstruction_mse, shared_frames, summarize_arms
from .gates import SHORTLIST_ORDER, admit_on_train, dense_gate, representative_score
from .io import FITTED, LEGACY, LOCKED
from .quantization import ORIGINAL, Quantizer, Stage

SCALE_MULTIPLIERS = (0.6, 0.8, 1.0, 1.2, 1.5)
MU_VALUES = (0, 2, 8)
ALLOCATIONS = (ORIGINAL, (10, 8, 8, 5, 5, 4), (10, 10, 8, 5, 4, 4))
RGB_FINALISTS = 6
RANKS = {"A2": 24, "A4-fsq": 36, "A5-fsq": 48}


def legacy_inputs(identity, device):
    lock = json_at(LOCKED)
    frozen = torch.load(FITTED, weights_only=True, map_location=device)
    if lock["fit_sha256"] != file_hash(FITTED) or frozen["identity"] != identity or lock["identity"] != identity:
        raise RuntimeError("sealed C1 artifacts do not match the frozen codec")
    fits = load_fits(frozen)
    anchor, baseline = {}, {"A0": {}, "A1": {}}
    for kind, space in (("I", ALIGNED), ("P", lock["winner"]["p_coordinate"])):
        fit = fits[key(kind, space)]
        indices = fit.indices[:24]
        stage = Stage([ORIGINAL] * 4, fit.scales[:4])
        baseline["A0"][kind] = Layout(indices, CHANNELS, space)
        baseline["A1"][kind] = Layout(indices, CHANNELS, space, Quantizer(stage))
        anchor[kind] = fit
    return lock, anchor, baseline


def sealed_training_plans(train, limits):
    plans = {}
    for seed in limits["seed_order"]:
        expected = json_at(LEGACY / f"c1a/seed{seed}_training.json")["plan"]
        actual = make_plans(train, Run("c1a", seed, limits["final_steps"], 4))
        if actual != expected:
            raise RuntimeError(f"seed {seed} changed the original C1-A Train order")
        plans[seed] = actual
    return plans


def used_frame_hashes(train, windows):
    return {
        item["sha256"] for video, start, length in windows for item in train.rows[video]["frame_hashes"][start:start + length]
    }


def calibration_plan(train, original_fit, seed_plans, excluded_videos):
    if train.split != "train":
        raise ValueError("calibration requires Train")
    train_videos = sorted(train.rows)
    occupied = [(row["video"], row["start"], row["frames"]) for row in original_fit]
    occupied.extend(
        (train_videos[row["source"]], row["start"], row["length"])
        for plan in seed_plans.values() for row in plan
    )
    used = used_frame_hashes(train, occupied)
    plan = []
    for video in sorted(set(train.rows) - set(excluded_videos)):
        hashes = train.rows[video]["frame_hashes"]
        groups = groupby(range(len(hashes)), key=lambda index: hashes[index]["sha256"] not in used)
        runs = [list(run)[:4] for available, run in groups if available]
        window = max(runs, key=len, default=[])
        if len(window) >= 2:
            plan.append({"video": video, "start": window[0], "frames": len(window)})
    if len(plan) < 5:
        raise RuntimeError("insufficient unused Train clips for disjoint calibration")
    return plan


def candidate_levels(importance):
    groups = len(importance)
    yield "legacy_allocation", (ORIGINAL,) * groups
    order = importance.argsort(dim=-1, descending=True, stable=True).tolist()
    for index, allocation in enumerate(ALLOCATIONS):
        ranked = sorted(allocation, reverse=True)
        yield f"sensitivity_{index}", tuple(
            tuple(ranked[channel_order.index(dim)] for dim in range(6)) for channel_order in order
        )


def proxy_ranking(values, energy):
    scale = fit_scale(values.reshape(-1, values.shape[1] * 6)).reshape(-1, 6)
    importance = values.var((0, 2, 3), correction=0) * energy
    ranked = []
    for (allocation, levels), multiplier, mu in product(candidate_levels(importance), SCALE_MULTIPLIERS, MU_VALUES):
        stage = Stage(levels, scale * multiplier, mu)
        ids, _ = stage.encode(values)
        error = (stage.decode(ids) - values).square()
        proxy = (error * energy[:, None, None, :]).mean().item()
        ranked.append((proxy, stage, {"allocation": allocation, "scale_multiplier": multiplier, "mu": mu}))
    return sorted(ranked, key=lambda trial: trial[0])


def train_samples(reader, train, codec, plan, baseline):
    samples = []
    energy = defaultdict(list)
    for item in plan:
        video = item["video"]
        clip = reader.read(train, video, item["start"], item["frames"])
        for sample, elapsed in shared_frames(codec, clip, baseline["A1"]):
            samples.append((video, "train_calibration", sample, elapsed))
            if sample.context.frame_id in (0, len(clip) - 1):
                space = baseline["A1"][sample.frame_type].space
                energy[sample.frame_type].append(sample_gradient(sample, codec.backbone)[key(sample.frame_type, space)])
        print(f"Train holdout {video} ({len(clip)} frames)", flush=True)
    return samples, {kind: torch.stack(values).mean(0).float() for kind, values in energy.items()}


def expanded_order(base, sensitivity):
    tail = sensitivity.argsort(descending=True, stable=True)
    return torch.cat((base, tail[~torch.isin(tail, base)]))


def eligible_rows(samples, space):
    return torch.cat([
        channel_rows(coordinate_value(sample.native, sample.aligned, space))
        for sample in samples
    ])


def select_stage(name, samples, layout, rows, energy, backbone, primary=None):
    expanded = rows.index_select(1, layout.indices).reshape(-1, layout.groups, 1, 1, 6)
    grouped_energy = energy[layout.indices].reshape(-1, 6)
    if primary is not None:
        ids, _ = primary.encode(expanded)
        expanded = expanded - primary.decode(ids)
    ranked = proxy_ranking(expanded, grouped_energy / grouped_energy.mean().clamp_min(1e-30))
    evaluated = []
    for proxy, stage, candidate in ranked[:RGB_FINALISTS]:
        quantizer = Quantizer(stage) if primary is None else Quantizer(primary, stage)
        case = Layout(layout.indices, layout.channels, layout.space, quantizer)
        rgb_mse = sum(reconstruction_mse(backbone, sample, case) for sample in samples) / len(samples)
        evaluated.append((rgb_mse, case, {**candidate, "proxy_mse": proxy, "rgb_mse": rgb_mse}))
    evaluated.sort(key=lambda trial: trial[0])
    best, case, choice = evaluated[0]
    print(f"Train {name} {samples[0].frame_type} MSE {best:.7f}, {choice}", flush=True)
    grid = [{**candidate, "proxy_mse": proxy} for proxy, _, candidate in ranked]
    return case, {"selected": choice, "rgb_finalists": [trial[2] for trial in evaluated], "grid": grid}


def fit_frame_type(kind, samples, energy, base, backbone):
    order = expanded_order(base.indices, energy)
    rows = eligible_rows(samples, base.space)
    arms, searches = {}, {}
    for name, rank in RANKS.items():
        cont = Layout(order[:rank], base.channels, base.space)
        arms[name], searches[name] = select_stage(name, samples, cont, rows, energy, backbone)
        if rank > 24:
            arms[name.replace("fsq", "cont")] = cont
    arms["A3"], searches["A3"] = select_stage("A3", samples, base, rows, energy, backbone, arms["A2"].quantizer.primary)
    return arms, searches


def fit_arms(backbone, samples, sensitivity, baseline):
    arms, searches = dict(baseline), {}
    for kind, energy in sensitivity.items():
        cases = [sample for _, _, sample, _ in samples if sample.frame_type == kind]
        fitted, searches[kind] = fit_frame_type(kind, cases, energy, baseline["A0"][kind], backbone)
        for name, layout in fitted.items():
            arms.setdefault(name, {})[kind] = layout
    return arms, searches


def calibrate(reader, train, codec, suite, legacy_lock, baseline, plans, excluded_videos, limits):
    plan = calibration_plan(train, legacy_lock["fit_plan"], plans, excluded_videos)
    samples, sensitivity = train_samples(reader, train, codec, plan, baseline)
    arms, searches = fit_arms(codec.backbone, samples, sensitivity, baseline)
    histograms = Histograms()
    rows = list(evaluate_samples(codec, samples, arms, suite, histograms))
    summaries = summarize_arms(rows)
    scores = {arm: summary["all"] for arm, summary in summaries.items()}
    candidates = admit_on_train(scores, limits)
    evidence = {
        "plan": plan, "frame_counts": reader.counts.copy(), "sensitivity": {k: v.tolist() for k, v in sensitivity.items()},
        "searches": searches, "shortlist": candidates,
        "arms": {arm: {kind: layout.record() for kind, layout in cases.items()} for arm, cases in arms.items()},
        "candidate_gates": {arm: dense_gate(scores["A1"], scores[arm], arm, limits) for arm in SHORTLIST_ORDER},
        "representative_score": {arm: representative_score(scores["A1"], scores[arm], limits) for arm in ("A3", "A5-fsq")},
        "summaries": summaries, "histograms": histograms.record(),
    }
    return rows, evidence
