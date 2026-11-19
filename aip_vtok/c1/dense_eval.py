"""Paired-input and independent closed-loop comparisons at the dense diagnostic budget."""

import torch
from torch.nn import functional as F

from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.diagnostics.diag0.io import Catalog
from aip_vtok.evaluation.fix import rms
from aip_vtok.tokenizer.fsq import FSQ

from .artifacts import THRESHOLD_DB, group_summary, write_rows
from .calibrate import key
from .diagnostic import decode_metrics, p_space
from .frames import capture, frame_fields
from .metrics import FrameMetrics, metric_suite
from .recovery_net import commit_reference
from .scan import DENSE
from .training import SEEDS, dense_observations, predict

PAIRED = "same_context_b0_fsq_history"
INDEPENDENT = "independent_closed"
BASELINE = "B0-fsq"
TRAINED = "B1-fsq"
DIAGNOSTIC = "B1-cont-diag"


def validation_videos(config, catalog):
    protocol = config["stage1"]["validation"]
    counts = {video: min(catalog.rows[video]["original_frames"], protocol["max_frames_per_video"])
              for video in protocol["videos"]}
    if sum(counts.values()) != protocol["total_frames"]:
        raise RuntimeError("validation cohort differs from the sealed 174-frame inventory")
    return counts


def dense_latent(backbone, model, sample, fit, space, obs):
    return predict(backbone, model, obs, fit, space, None, sample.context)


def observed(sample, lock, fits, fsq):
    space = p_space(sample, lock)
    fit = fits[key(sample.frame_type, space)]
    return fit, space, dense_observations(fsq, sample, fit, space)


def dense_row(video, mode, sample, scorer, backbone, latent, method, seed=0, **diagnostics):
    return {**frame_fields(sample, video, mode, "validation"),
            "method": method, "seed": seed, "budget": DENSE, "diagnostic_only": method == DIAGNOSTIC,
            **decode_metrics(backbone, sample, scorer, latent), **diagnostics}


def commit(codec, sample, latent):
    projected = codec.projector(latent, sample.context.intra)
    return commit_reference(sample.context, latent, projected)


def ref_details(state, sample):
    return {"reference_budget": DENSE, "reference_latent_rms": rms(state.latent),
            "reference_latent_mse": F.mse_loss(state.latent, sample.aligned).item(),
            "projected_feature_rms": rms(state.feature)}


def paired_frame(codec, sample, scorer, video, fits, lock, fsq, models):
    fit, space, quantized = observed(sample, lock, fits, fsq)
    cont = dense_observations(fsq, sample, fit, space, quantized=False)
    backbone = codec.backbone
    zero = dense_latent(backbone, None, sample, fit, space, quantized)
    state = commit(codec, sample, zero)
    baseline = dense_row(video, PAIRED, sample, scorer, backbone, zero, BASELINE, **ref_details(state, sample))
    rows = [baseline, baseline | {"history": INDEPENDENT}]
    for method, obs, seed, model in (
        ("B0-cont", cont, 0, None),
        *((name, obs, seed, model) for seed, model in models.items()
          for name, obs in ((TRAINED, quantized), (DIAGNOSTIC, cont))),
    ):
        latent = dense_latent(backbone, model, sample, fit, space, obs)
        rows.append(dense_row(video, PAIRED, sample, scorer, backbone, latent, method, seed))
    return state, rows


def independent_frame(codec, image, index, state, scorer, video, fits, lock, fsq, model, seed):
    sample = capture(codec.backbone, state, index, image)
    fit, space, obs = observed(sample, lock, fits, fsq)
    latent = dense_latent(codec.backbone, model, sample, fit, space, obs)
    next_state = commit(codec, sample, latent)
    row = dense_row(video, INDEPENDENT, sample, scorer, codec.backbone, latent, TRAINED, seed,
                    **ref_details(next_state, sample))
    return next_state, row


@torch.no_grad()
def dense_video(codec, clip, video, suite, fits, lock, models):
    fsq = FSQ().cuda()
    baseline = ReferenceState()
    states = {seed: ReferenceState() for seed in models}
    rows = []
    for frame, image in enumerate(clip):
        sample = capture(codec.backbone, baseline, frame, image)
        scorer = FrameMetrics(suite, sample.rgb)
        baseline, same_context = paired_frame(codec, sample, scorer, video, fits, lock, fsq, models)
        rows.extend(same_context)
        for seed, model in models.items():
            states[seed], row = independent_frame(
                codec, image, frame, states[seed], scorer, video, fits, lock, fsq, model, seed,
            )
            rows.append(row)
    return rows


def dense_decision(summary, expected_frames):
    table = {(r["history"], r["method"], r["seed"]): r for r in summary}
    gains = []
    for mode in (PAIRED, INDEPENDENT):
        b0 = table[mode, BASELINE, 0]
        for seed in SEEDS:
            b1 = table[mode, TRAINED, seed]
            if b0["frames"] != expected_frames or b1["frames"] != expected_frames:
                raise ValueError("dense gate requires the full paired validation cohort for both seeds")
            psnr_gain = b1["psnr_global_mse"] - b0["psnr_global_mse"]
            lpips_gain = b0["lpips"] - b1["lpips"]
            gains.append({"seed": seed, "history": mode, "psnr_gain_db": psnr_gain, "lpips_gain": lpips_gain,
                          "dists_gain": b0["dists"] - b1["dists"],
                          "passed": psnr_gain >= THRESHOLD_DB and lpips_gain > 0})
    return gains, all(row["passed"] for row in gains)


def evaluate_dense(config, codec, fits, lock, models):
    catalog = Catalog.open(config, "validation")
    counts = validation_videos(config, catalog)
    suite = metric_suite()
    rows = []
    for video, count in counts.items():
        rows.extend(dense_video(codec, catalog.read(video, 0, count), video, suite, fits, lock, models))
        print(f"C1-A {video} validation {count} frames per arm", flush=True)
    keys = ("history", "method", "seed")
    write_rows("c1a/validation", rows, keys, by_frame_type=group_summary(rows, (*keys, "frame_type")))
    return dense_decision(group_summary(rows, keys), sum(counts.values()))
