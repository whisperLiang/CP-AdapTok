"""Matched 174-frame and independent 64-frame sparse comparisons."""

from itertools import product

import torch
from torch.nn import functional as F

from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.diagnostics.diag0.closed import FIX_A as FIX_A_LAYOUT
from aip_vtok.diagnostics.diag0.io import Catalog, json_at
from aip_vtok.evaluation.fix import HORIZON_VIDEO, rms
from aip_vtok.runtime import file_hash
from aip_vtok.tokenizer.fsq import FSQ
from aip_vtok.training.prefix import prefix_latents

from .artifacts import DIAG0, OUTPUT, group_summary, write_rows
from .calibrate import GAUSS, ZERO
from .dense_eval import validation_videos
from .diagnostic import HORIZON, decode_metrics, rate, sparse_rollout
from .frames import capture, frame_fields
from .metrics import FrameMetrics, metric_suite, quality_backslides
from .recovery_net import commit_reference
from .scan import BUDGETS, REF_BUDGET
from .training import predict, prepare_frame

RECOVERED = "B3"
FIX_A = "Fix-A"
MATCHED = "same_context_b2_history"
SCORE_KEYS = ("method", "seed", "budget")


def reference_values(codec, sample, latent, budget):
    feature = codec.projector(latent, sample.context.intra)
    return feature, {"reference_budget": budget, "reference_latent_rms": rms(latent),
                     "reference_latent_mse": F.mse_loss(latent, sample.aligned).item(),
                     "projected_feature_rms": rms(feature)}


def paired_sparse(codec, sample, scorer, fields, frame, model, method, seed, sigma):
    ref = REF_BUDGET[sample.context.intra]
    width = None if method == ZERO else sigma
    receiver = (predict(codec.backbone, model, frame.read(k), frame.fit, frame.space, width, sample.context)
                for k in BUDGETS)
    latents = dict(zip(BUDGETS, receiver))
    feature, reference = reference_values(codec, sample, latents[ref], ref)
    rows = [
        {**fields, "method": method, "seed": seed, "budget": k, **reference,
         **decode_metrics(codec.backbone, sample, scorer, aligned),
         **rate(k, max(k, ref), sample.rgb.shape[-2:].numel(), frame.payload)}
        for k, aligned in latents.items()
    ]
    return latents[ref], feature, rows


def paired_fix(codec, sample, state, scorer, fields):
    encoded = codec.encode_frame(sample.rgb, state, sample.context.frame_id)
    latents = {k: prefix_latents(codec, encoded.tokens, encoded.prediction, k, differentiable=False)
               for k in BUDGETS}
    ref = REF_BUDGET[sample.context.intra]
    _, reference = reference_values(codec, sample, latents[ref][0], ref)
    return [{**fields, "method": FIX_A, "seed": 0, "budget": k, **reference,
             **decode_metrics(codec.backbone, sample, scorer, fused)}
            for k, (_, fused) in latents.items()]


@torch.no_grad()
def matched_video(clip, video, codec, fits, lock, suite, models):
    state = ReferenceState()
    slots = torch.tensor(lock["slots"], device="cuda")
    fsq = FSQ().cuda()
    cases = [(GAUSS, 0, None), (ZERO, 0, None), *[(RECOVERED, seed, model) for seed, model in models.items()]]
    for frame_id, image in enumerate(clip):
        sample = capture(codec.backbone, state, frame_id, image)
        scorer = FrameMetrics(suite, sample.rgb)
        frame = prepare_frame(sample, fits, lock, fsq, slots)
        fields = frame_fields(sample, video, MATCHED, "validation")
        yield from paired_fix(codec, sample, state, scorer, fields)
        for method, seed, model in cases:
            latent, feature, rows = paired_sparse(
                codec, sample, scorer, fields, frame, model, method, seed, lock["winner"]["sigma"],
            )
            yield from rows
            if method == GAUSS:
                state = commit_reference(sample.context, latent, feature)


def comparisons(rows, models, frames):
    summary = group_summary(rows, SCORE_KEYS)
    table = {(r["method"], r["seed"], r["budget"]): r for r in summary}
    for seed, k, method in product(models, BUDGETS, (GAUSS, ZERO, FIX_A)):
        b3 = table[RECOVERED, seed, k]
        b = table[method, 0, k]
        if min(b3["frames"], b["frames"]) != frames or b3["frames"] != b["frames"]:
            raise RuntimeError("sparse comparison cohort is incomplete")
        yield {"seed": seed, "budget": k, "baseline": method,
               "psnr_gain_db": b3["psnr_global_mse"] - b["psnr_global_mse"],
               "lpips_gain": b["lpips"] - b3["lpips"], "dists_gain": b["dists"] - b3["dists"],
               "reference_latent_mse_gain": b["reference_latent_mse"] - b3["reference_latent_mse"]}


def store(name, rows, models, frames, **metadata):
    gains = list(comparisons(rows, models, frames))
    write_rows(f"c1b/{name}", rows, SCORE_KEYS, gains=gains,
               by_frame_type=group_summary(rows, (*SCORE_KEYS, "frame_type")),
               quality_backslides=quality_backslides(rows, ("method", "seed", "frame_type")),
               **metadata)
    return gains


def sealed_baselines():
    diag = OUTPUT / "diag05/closed.json"
    fix = DIAG0 / "d0_e_closed_loop_64f.json"
    native_c1 = [r | {"seed": 0} for r in json_at(diag)["rows"]]
    prior = [r | {"seed": 0, "method": FIX_A, "history": "independent_closed",
                  "source": "Diag-0 exact closed-loop replay of caae55b"}
             for r in json_at(fix)["rows"] if r["layout"] == FIX_A_LAYOUT]
    return native_c1 + prior, {"diag05": file_hash(diag), "fix_a_caae55b": file_hash(fix)}


def evaluate_sparse(config, codec, fits, lock, models):
    catalog = Catalog.open(config, "validation")
    suite = metric_suite()
    counts = validation_videos(config, catalog)
    rows = []
    for video, count in counts.items():
        rows.extend(matched_video(catalog.read(video, 0, count), video, codec, fits, lock, suite, models))
        print(f"C1-B {video} matched validation {count} frames per arm", flush=True)
    paired = store("validation", rows, models, sum(counts.values()), history=MATCHED, frame_counts=counts)
    clip = catalog.read(HORIZON_VIDEO, 0, HORIZON)
    rows, sources = sealed_baselines()
    for seed, model in models.items():
        rows.extend(row | {"seed": seed} for row in sparse_rollout(
            clip, HORIZON_VIDEO, codec, suite, lock, fits, RECOVERED, model,
        ))
        print(f"C1-B seed={seed} independent rollout {HORIZON} frames", flush=True)
    closed = store("closed", rows, models, HORIZON, history="independent_closed", baseline_sources_sha256=sources)
    return paired, closed
