"""Compare all five receivers on common packets and on independent histories."""

from collections import defaultdict
from itertools import product

import torch
from PIL import Image, ImageDraw, ImageFont

from aip_vtok.c1.scan import BUDGETS
from aip_vtok.diagnostics.diag0.io import json_at
from aip_vtok.evaluation.fix import HORIZON_VIDEO
from aip_vtok.runtime import file_hash, write_json
from aip_vtok.sparse_rfsq.evaluation import summaries
from aip_vtok.sparse_rfsq.gates import pilot_decision
from aip_vtok.sparse_rfsq.protocol import FOCUS, LIMITS, UPDATES
from aip_vtok.sparse_rfsq.receiver import B2, B3, Receiver
from aip_vtok.sparse_rfsq.rollout import CausalPair

from .evaluation import (
    CLOSED,
    FIXED,
    FrameScorer,
    display_rows,
    frame_fields,
    validate_rows,
    validation,
    write_csv,
)
from .models import ARMS, DecoderReceiver, create_model
from .phase1 import TESTS, require_phase1
from .protocol import OLD, OUTPUT, PROVENANCE, SIGMA
from .training import PLAN, checkpoint_path, verify_checkpoints

METHODS = {"E0": "B2", "E1": "W1", **ARMS}
EVIDENCE = ("pilot_per_frame.csv", "pilot_metrics.csv", "pilot_gates.json", "pilot_reconstruction.png")
COMPLETE = OUTPUT / "pilot_evaluation.json"


def completed_training():
    require_phase1()
    results = {arm: json_at(OUTPUT / f"pilot_train_{arm}.json") for arm in ARMS}
    for arm, result in results.items():
        identity = result["identity"]
        if (result["steps"] != UPDATES or not result["integrity"]["passed"]
                or identity["arm"] != arm or identity["provenance_sha256"] != file_hash(PROVENANCE)
                or identity["tested_source_sha256"] != file_hash(TESTS) or identity["plan_sha256"] != file_hash(PLAN)
                or result["integrity"]["images_read"]["validation"] or result["integrity"]["images_read"]["test"]):
            raise RuntimeError(f"{arm} pilot needs a verified 500-step checkpoint from the locked Train schedule")
        verify_checkpoints(arm, result)
    return results


def receivers(run, records):
    loaded = {"E0": Receiver(SIGMA), "E1": Receiver(SIGMA, run.w1)}
    for arm in records:
        checkpoint = torch.load(checkpoint_path(arm, UPDATES), weights_only=True, map_location="cpu")
        model = create_model(arm, "cuda")
        model.load_state_dict(checkpoint["model"], strict=True)
        loaded[arm] = DecoderReceiver(SIGMA, model.eval().requires_grad_(False))
    return loaded


class Reconstructions:
    """Native RGB crop at the midpoint of the declared late-frame interval."""

    VIDEO, FRAME, BUDGETS = HORIZON_VIDEO, 55, FOCUS
    X, Y, WIDTH, HEIGHT = 768, 432, 384, 216

    def __init__(self, backbone):
        self.backbone = backbone
        self.images = {}

    def crop(self, rgb):
        crop = rgb[0, :, self.Y:self.Y + self.HEIGHT, self.X:self.X + self.WIDTH].permute(1, 2, 0)
        return Image.fromarray(crop.mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy())

    def __call__(self, frame, latent, row):
        budget = row["budget"]
        if row["video"] != self.VIDEO or row["frame"] != self.FRAME or budget not in self.BUDGETS:
            return
        original = budget, "Original"
        if original not in self.images:
            self.images[original] = self.crop(frame.sample.rgb)
        rec, _ = self.backbone.decode_aligned(latent, frame.context)
        self.images[budget, row["arm"]] = self.crop(rec)

    def publish(self):
        labels = {"Original": "Original", **METHODS}
        canvas = Image.new("RGB", (self.WIDTH * len(labels), (self.HEIGHT + 30) * len(self.BUDGETS)), "#171717")
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.truetype("DejaVuSans.ttf", 17)
        for index, budget in enumerate(self.BUDGETS):
            y = index * (self.HEIGHT + 30)
            for col, (arm, name) in enumerate(labels.items()):
                x = col * self.WIDTH
                draw.text((x + 12, y + 4), f"{name}   K{budget}", fill="white", font=font)
                canvas.paste(self.images[budget, arm], (x, y + 30))
        canvas.save(OUTPUT / "pilot_reconstruction.png")
        return {"video": self.VIDEO, "frame_zero_based": self.FRAME, "history": FIXED,
                "crop_xywh": [self.X, self.Y, self.WIDTH, self.HEIGHT], "color": "8-bit RGB, exact native crop"}


@torch.no_grad()
def pilot_video(run, catalog, clip, video, cases, models, visualization):
    shared = CausalPair()
    independent = {arm: CausalPair() for arm in models}
    backbone = run.codec.backbone
    for index, image in enumerate(clip):
        fixed = shared.advance(run.codec, image, index, run.inputs.legacy, Receiver(SIGMA))
        fixed_common = frame_fields(catalog, video, fixed, shared, "C1-A1")
        fixed_packet = cases[fixed.sample.frame_type].encode(fixed.sample)
        scorer = FrameScorer(run.codec, run.suite, fixed.sample.rgb)
        for arm, receive in models.items():
            pair = independent[arm]
            yield from display_rows(scorer, backbone, fixed, cases[fixed.sample.frame_type], fixed_packet,
                                    receive, fixed_common | {"arm": arm, "history": FIXED}, BUDGETS, visualization)
            closed = pair.advance(run.codec, image, index, cases, receive)
            info = frame_fields(catalog, video, closed, pair, arm)
            yield from display_rows(scorer, backbone, closed, cases[closed.sample.frame_type], closed.payload,
                                    receive, info | {"arm": arm, "history": CLOSED}, BUDGETS)
        del scorer


def validate_fixed(rows):
    seen = defaultdict(set)
    fields = ("input_rgb_sha256", "aligned_sha256", "reference_state_id", "reference_payload_sha256", "payload_sha256")
    for row in rows:
        if row["history"] == FIXED:
            seen[row["video"], row["frame"], row["budget"]].add(tuple(row[k] for k in fields))
    if any(len(identities) != 1 for identities in seen.values()):
        raise RuntimeError("FIXED receivers got different inputs, anchors, or integer prefixes")
    return len(seen)


def gate_rows(rows):
    for row in rows:
        baseline = row["arm"] == "E0"
        yield row | {"strategy": "SP-1", "method": B2 if baseline else B3,
                     "variant": "base" if baseline else row["arm"],
                     "history": row["history"] if baseline else row["history"] + "_B3"}


def metric_rows(summary):
    for scope, points in (("validation", summary["validation"]["points"]), ("horizon", summary["horizon"]["points"]),
                          ("late", summary["late"])):
        for p in points:
            arm = "E0" if p["method"] == B2 else p["variant"]
            for kind in ("all", "I", "P"):
                if kind in p:
                    yield {"scope": scope, "arm": arm, "network": METHODS[arm], "history": p["history"].removesuffix("_B3"),
                           "budget": p["budget"], "frame_type": kind, **p[kind]}


def gate_margins(decision):
    checks, gains = decision["checks"], decision["gains"]
    focus = [r for r in gains["validation"] if r["history"] == CLOSED and r["budget"] in FOCUS]
    late = [r for r in gains["late"] if r["history"] == CLOSED]
    backslides = {scope: [r | {"margin_db": r["limit_db"] - r["observed_db"]} for r in group]
                  for scope, group in decision["psnr_backslide_checks"].items()}
    threshold = LIMITS["focus_min_psnr_gain_db"]
    worst_psnr = [min(r["psnr_gain_db"], r["mean_frame_psnr_gain_db"]) for r in late]
    minimum_psnr = LIMITS["late_min_psnr_gain_db"]
    psnr = [{"budget": r["budget"], "observed_db": value, "limit_db": minimum_psnr,
             "margin_db": value - minimum_psnr} for r, value in zip(late, worst_psnr, strict=True)]
    limits = {"lpips": LIMITS["late_max_lpips_increase"], "dists": LIMITS["late_max_dists_increase"]}
    perception = [{"budget": r["budget"], **{name + "_margin": limit - r[name + "_change"]
                                             for name, limit in limits.items()}} for r in late]
    rms = [{"budget": r["budget"], "observed": r["rms"], "baseline": r["baseline_rms"],
            "limit": r["baseline_rms"] * LIMITS["late_max_rms_relative_to_b2"],
            "margin": r["baseline_rms"] * LIMITS["late_max_rms_relative_to_b2"] - r["rms"]} for r in late]
    margins = {
        "G1": {"passed": checks["focus_psnr"], "threshold_db": threshold,
               "positive_margins_db": {r["budget"]: r["psnr_gain_db"] for r in focus},
               "target_margins_db": {r["budget"]: r["psnr_gain_db"] - threshold for r in focus}},
        "G2": {"passed": checks["focus_lpips_dists"],
               "by_budget": [{"budget": r["budget"], "lpips_margin": -r["lpips_change"],
                              "dists_margin": LIMITS["focus_max_dists_increase"] - r["dists_change"]} for r in focus]},
        "G3": {"passed": all(r["margin_db"] >= 0 for r in psnr), "by_budget": psnr},
        "G4": {"passed": all(min(r["lpips_margin"], r["dists_margin"]) >= 0 for r in perception),
               "by_budget": perception},
        "G5": {"passed": checks["reference_rms"], "maximum_horizon_rms": decision["max_reference_latent_rms"],
               "absolute_margin": LIMITS["max_reference_latent_rms"] - decision["max_reference_latent_rms"],
               "late_relative": rms},
        "G6": {"passed": checks["psnr_backslides"], "scopes": backslides},
        "G7": {"passed": checks["hard_integrity"]},
    }
    if all(g["passed"] for g in margins.values()) != decision["passed"]:
        raise RuntimeError("gate margin rendering disagrees with the original S2 decision")
    return margins


def decisions(rows, integrity):
    base = summaries(list(gate_rows([r for r in rows if r["arm"] == "E0"])))
    result = {}
    for arm in ARMS:
        selected = [r for r in rows if r["arm"] == arm]
        learned = summaries(list(gate_rows(selected)))
        max_rms = max(r["reference_latent_rms"] for r in selected
                      if r["history"] == CLOSED and r["video"] == HORIZON_VIDEO)
        decision = pilot_decision(base, learned, arm, max_rms, integrity["passed"])
        result[arm] = decision | {"gates": gate_margins(decision)}
    return result


def replay_controls(rows):
    def get_key(row):
        return row["video"], row["frame"], row["budget"], row["history"].removesuffix("_B3")

    mapping = {"E0": ("s1", "base", B2), "E1": ("s2", "W1", B3)}
    replay = {}
    for arm, (stage, variant, method) in mapping.items():
        old = {get_key(row): row for row in json_at(OLD / f"{stage}.json")["rows"]
               if (row["strategy"], row["variant"], row["method"]) == ("SP-1", variant, method)}
        new = {get_key(row): row for row in rows if row["arm"] == arm}
        if new.keys() != old.keys():
            raise RuntimeError(f"{arm} cohort differs from the sealed S2 control")
        replay[arm] = {"frames_and_budgets": len(new), "max_abs_difference": {
            name: max(abs(row[name] - old[key][name]) for key, row in new.items())
            for name in ("mse", "psnr", "lpips", "dists", "latent_mse", "reference_latent_rms")
        }}
    return replay


def run_evaluation(run):
    training = completed_training()
    if COMPLETE.is_file():
        audit = json_at(COMPLETE)
        if audit["outputs"] != {name: file_hash(OUTPUT / name) for name in EVIDENCE}:
            raise RuntimeError("final evaluation files changed after publication")
        print("Final evaluation already verified.", flush=True)
        return
    models = receivers(run, training)
    cases = run.inputs.cases("SP-1", run.lock["keys"]["SP-1"])
    catalog, counts = validation(run)
    visualization = Reconstructions(run.codec.backbone)
    rows = []
    for video, count in counts.items():
        clip = run.reader.read(catalog, video, 0, count)
        rows.extend(pilot_video(run, catalog, clip, video, cases, models, visualization))
        print(f"Pilot evaluation · {video} · {count} frames · {len(rows)} cumulative measurements", flush=True)
    validate_rows(rows, counts, set(product(METHODS, (FIXED, CLOSED), BUDGETS)))
    fixed_matches = validate_fixed(rows)
    integrity = run.verify_integrity() | {"verified_fixed_frame_budget_payloads": fixed_matches}
    summary = summaries(list(gate_rows(rows)))
    write_csv(OUTPUT / "pilot_per_frame.csv", rows)
    write_csv(OUTPUT / "pilot_metrics.csv", list(metric_rows(summary)))
    write_json(OUTPUT / "pilot_gates.json", {"limits": LIMITS, "decisions": decisions(rows, integrity),
                                           "integrity": integrity})
    crop = visualization.publish()
    write_json(COMPLETE, {
        "rows": len(rows), "cohort": counts, "reconstruction": crop, "provenance_sha256": file_hash(PROVENANCE),
        "training_sha256": {arm: file_hash(OUTPUT / f"pilot_train_{arm}.json") for arm in training},
        "sealed_controls_replay": replay_controls(rows),
        "outputs": {name: file_hash(OUTPUT / name) for name in EVIDENCE},
    })
