"""Select one static scan and the shared interpolator on independent Train videos."""

from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import product
from statistics import mean

import numpy as np
import torch
from torch.nn import functional as F

from aip_vtok.c1.calibrate import key, sample_gradient
from aip_vtok.c1.scan import BUDGETS, GROUPS, SHAPE, SIGMAS
from aip_vtok.c1_followup.calibration import legacy_inputs, used_frame_hashes
from aip_vtok.c1_followup.experiment import frozen_candidates, load_layouts
from aip_vtok.c1_followup.io import OUTPUT as FOLLOWUP
from aip_vtok.c1_followup.io import create_record, write_rows
from aip_vtok.diagnostics.diag0.io import json_at
from aip_vtok.evaluation.gates import psnr
from aip_vtok.runtime import file_hash

from .protocol import (
    CALIBRATION,
    CALIBRATION_LENGTH,
    CALIBRATION_SEED,
    CALIBRATION_VIDEOS,
    MIN_POSITIVE_STEPS,
    PROTOCOL,
    SCAN_LOCK,
    timestamp,
)
from .receiver import Receiver
from .rollout import CausalPair, FrameCode
from .scan import POLICIES, REFINEMENT_PCTS, Plan, rates, scan_table

KINDS = ("I", "P")
TRIALS = (("SP-1", 0), ("SP-2", 0), *(("SP-3", pct) for pct in REFINEMENT_PCTS))


def codes(keys, layouts, entropy):
    return {kind: FrameCode(layout, Plan.create(keys[kind], SHAPE, GROUPS, layout.indices.device),
                            rates(keys[kind], entropy[kind])) for kind, layout in layouts.items()}


@dataclass(frozen=True)
class Inputs:
    layouts: dict
    entropy: dict
    fits: dict
    legacy: dict
    legacy_sigma: float
    excluded: set

    def cases(self, strategy, keys):
        arm = "A2" if strategy == "SP-1" else "A3"
        return codes(keys, self.layouts[arm], self.entropy[arm])


def check_layouts(layouts, old):
    for kind in KINDS:
        a1, a2, a3 = (layouts[arm][kind].record() for arm in ("A1", "A2", "A3"))
        if a1 != old[kind].record() or a2 != a3 | {"stages": a3["stages"][:1]}:
            raise RuntimeError(f"{kind} frozen A1 replay or A2/A3 primary mismatch")
        if a1["space"] != "aligned" or a2["space"] != "aligned" or a1["indices"] != a2["indices"]:
            raise RuntimeError("sparse recovery requires the shared, aligned 24-channel layout")


def load_inputs(config, identity):
    lock, fits, old = legacy_inputs(identity, "cuda")
    frozen = frozen_candidates()
    layouts = load_layouts({"arms": {arm: frozen["arms"][arm] for arm in ("A1", "A2", "A3")}}, "cuda")
    check_layouts(layouts, old["A1"])
    data = json_at(FOLLOWUP / "calibration.json")
    entropy = read_entropy(layouts, data["histograms"])
    excluded = set(config["calibration"]["videos"]) | {
        row["video"] for row in lock["calibration_plan"] + data["plan"]
    }
    area = SHAPE[0] * SHAPE[1]
    keys = tuple((1, slot // area, *divmod(slot % area, SHAPE[1])) for slot in lock["slots"])
    legacy = codes(dict.fromkeys(KINDS, keys), old["A1"], entropy["A1"])
    return Inputs(layouts, entropy, fits, legacy, lock["winner"]["sigma"], excluded)


def read_entropy(layouts, histograms):
    by_stage = {(r["arm"], r["frame_type"], r["level"]): [g["id_entropy_bits"] for g in r["groups"]]
                for r in histograms}
    return {arm: {kind: [by_stage[arm, kind, level] for level in range(1, len(layout.quantizer.stages) + 1)]
                  for kind, layout in cases.items()} for arm, cases in layouts.items()}


def holdout_plan(train, excluded):
    if train.split != "train":
        raise ValueError("scan calibration requires a Train catalog")
    eligible = sorted(video for video, row in train.rows.items()
                      if video not in excluded and row["original_frames"] >= CALIBRATION_LENGTH)
    rng = np.random.default_rng(CALIBRATION_SEED)
    selected = sorted(map(str, rng.choice(eligible, CALIBRATION_VIDEOS, replace=False)))
    return [{"video": video, "start": int(rng.integers(train.rows[video]["original_frames"] - CALIBRATION_LENGTH + 1)),
             "frames": CALIBRATION_LENGTH} for video in selected]


def training_catalog(train, holdout):
    videos = {row["video"] for row in holdout}
    held_hashes = used_frame_hashes(train, [(video, 0, train.rows[video]["original_frames"]) for video in videos])
    rows = {video: row for video, row in train.rows.items() if video not in videos and not any(
        frame["sha256"] in held_hashes for frame in row["frame_hashes"]
    )}
    if not rows or train.split != "train":
        raise ValueError("pilot needs Train video content independent of calibration")
    return replace(train, rows=rows)


def holdout_frames(run, train, plan):
    frames, energies = [], defaultdict(list)
    for row in plan:
        video = row["video"]
        pair = CausalPair()
        clip = run.reader.read(train, video, row["start"], row["frames"])
        for index, image in enumerate(clip):
            frame = pair.advance(run.codec, image, index, run.inputs.legacy, Receiver(run.inputs.legacy_sigma))
            sample = frame.sample
            frames.append((video, index + row["start"], frame))
            if index in (0, len(clip) - 1):
                kind = sample.frame_type
                layout = run.inputs.layouts["A2"][kind]
                gradient = sample_gradient(sample, run.codec.backbone)[key(kind, layout.space)]
                energies[kind].append(gradient[layout.indices].reshape(GROUPS, -1).sum(1))
        print(f"S1 Train holdout {video} {len(clip)} frames", flush=True)
    weights = {}
    for kind, entries in energies.items():
        amount = torch.stack(entries).mean(0)
        fractions = (1_000_000 * amount / amount.sum().clamp_min(1e-30)).tolist()
        weights[kind] = [max(1, round(x)) for x in fractions]
    return frames, weights


def trial_tables(weights):
    return {(name, pct): {kind: scan_table(name, group_weights, pct) for kind, group_weights in weights.items()}
            for name, pct in TRIALS}


def calibration_scores(run, frames, tables):
    for (strategy, pct), keys in tables.items():
        cases = run.inputs.cases(strategy, keys)
        for video, source_frame, frame in frames:
            yield from score_frame(run.codec.backbone, frame, cases, video, source_frame, strategy, pct)
        print(f"S1 Train grid {strategy} residual={pct}%", flush=True)


@torch.no_grad()
def score_frame(backbone, frame, cases, video, source_frame, strategy, pct):
    sample = frame.sample
    case = cases[sample.frame_type]
    payload = case.encode(sample)
    for budget, sigma in product(BUDGETS, SIGMAS):
        latent = Receiver(sigma)(case, backbone, frame.context, payload[:, :budget])
        rec, _ = backbone.decode_aligned(latent, frame.context)
        yield {"strategy": strategy, "refinement_pct": pct, "video": video, "source_frame": source_frame,
               "frame_type": sample.frame_type, "budget": budget, "sigma": sigma,
               "mse": F.mse_loss(rec, sample.rgb).item()}


def has_progress(curve):
    return curve[-1] > curve[0] and sum(b > a for a, b in zip(curve, curve[1:])) >= MIN_POSITIVE_STEPS


def score_curves(rows):
    errors = defaultdict(list)
    for r in rows:
        k = tuple(r[key] for key in ("strategy", "refinement_pct", "sigma", "budget", "frame_type"))
        errors[k].append(r["mse"])
    trials = {}
    for (strategy, pct), sigma in product(TRIALS, SIGMAS):
        curve = [mean(psnr(mean(errors[strategy, pct, sigma, k, kind])) for kind in KINDS) for k in BUDGETS]
        trials[strategy, pct, sigma] = {"strategy": strategy, "refinement_pct": pct, "sigma": sigma,
                                       "curves_db": curve, "score_db": mean(curve), "progressive": has_progress(curve)}
    return trials


def common_sigma_scores(trials):
    return {sigma: mean(mean(trial["score_db"] for trial in trials.values()
                             if trial["strategy"] == name and trial["sigma"] == sigma)
                         for name in POLICIES) for sigma in SIGMAS}


def select(rows):
    trials = score_curves(rows)
    sigma_scores = common_sigma_scores(trials)
    sigma = max(SIGMAS, key=sigma_scores.__getitem__)
    pct = max(REFINEMENT_PCTS, key=lambda r: trials["SP-3", r, sigma]["score_db"])
    finalists = [trials[name, pct if name == "SP-3" else 0, sigma] for name in POLICIES]
    ranked = sorted((row for row in finalists if row["progressive"]), key=lambda row: -row["score_db"])
    return {"sigma": sigma, "sp3_refinement_pct": pct, "winner": ranked[0]["strategy"] if ranked else None,
            "finalists": finalists, "trials": list(trials.values()),
            "sigma_scores": [{"sigma": s, "score_db": v} for s, v in sigma_scores.items()]}


def freeze_scan(run):
    if SCAN_LOCK.is_file():
        return load_scan()
    train = run.reader.open("train")
    plan = holdout_plan(train, run.inputs.excluded)
    create_record(CALIBRATION.with_name("holdout.json"), {"plan": plan, "protocol_sha256": file_hash(PROTOCOL)})
    frames, weights = holdout_frames(run, train, plan)
    tables = trial_tables(weights)
    rows = list(calibration_scores(run, frames, tables))
    chosen = select(rows)
    pct = chosen["sp3_refinement_pct"]
    keys = {name: tables[name, pct if name == "SP-3" else 0] for name in POLICIES}
    write_rows(CALIBRATION, rows, weights=weights, selection=chosen, holdout=plan)
    run.verify_integrity()
    record = {"locked_at": timestamp(), "protocol_sha256": file_hash(PROTOCOL),
              "calibration_sha256": file_hash(CALIBRATION.with_suffix(".json")), "holdout": plan, "keys": keys,
              "sigma": chosen["sigma"], "sp3_refinement_pct": pct, "winner": chosen["winner"],
              "validation_images_before_lock": run.reader.counts["validation"], "test_images_read": 0}
    create_record(SCAN_LOCK, record)
    print(f"S1 frozen on Train: {chosen['winner']}, sigma={chosen['sigma']}, SP-3 residual={pct}%", flush=True)
    return load_scan()


def load_scan():
    path = CALIBRATION.with_suffix(".json")
    lock = json_at(SCAN_LOCK)
    if (lock["protocol_sha256"] != file_hash(PROTOCOL) or lock["calibration_sha256"] != file_hash(path)
            or lock["validation_images_before_lock"] or lock["test_images_read"]):
        raise RuntimeError("frozen scan lacks Train-only provenance")
    evidence = json_at(path)
    validate_scan(lock, evidence)
    return lock


def validate_scan(lock, evidence):
    chosen = select(evidence["rows"])
    tables = trial_tables(evidence["weights"])
    if any(chosen[key] != lock[key] for key in ("sigma", "sp3_refinement_pct", "winner")):
        raise RuntimeError("selected schedule disagrees with Train calibration")
    for strategy in POLICIES:
        pct = lock["sp3_refinement_pct"] if strategy == "SP-3" else 0
        actual = {kind: tuple(map(tuple, keys)) for kind, keys in lock["keys"][strategy].items()}
        if actual != tables[strategy, pct]:
            raise RuntimeError("scan lock disagrees with deterministic Train schedule")
    if lock["holdout"] != evidence["holdout"]:
        raise RuntimeError("frozen Train holdout changed")
