"""Fit Train-only channel statistics and select a scan on disjoint Train videos."""

from dataclasses import replace
from itertools import product

import numpy as np
import torch
from torch.nn import functional as F

from aip_vtok.diagnostics.diag0.fit import CALIBRATION_STRIDE, FIT_SEED, training_plan
from aip_vtok.diagnostics.diag0.io import Catalog
from aip_vtok.diagnostics.diag0.matched import CLEAN, LOW
from aip_vtok.evaluation.gates import psnr
from aip_vtok.runtime import file_hash, save_checkpoint, write_json
from aip_vtok.tokenizer.fsq import FSQ

from .artifacts import FITTED, LOCKED, THRESHOLD_DB, timestamp, write_rows
from .channels import (
    ALIGNED,
    NATIVE,
    ChannelFit,
    aligned_value,
    coordinate_value,
    fit_channels,
    native_gradient,
)
from .frames import clean_samples, frame_fields, history_samples
from .scan import BUDGETS, SCHEDULES, SHAPE, SIGMAS, TOP, scan_schedule
from .tokens import VERSIONS, grouped, receive, send

CALIBRATION_VIDEOS = 5
FIT_SPACES = ("I_aligned", "P_aligned", "P_p_native")
ZERO = "B2-ZeroFill"
GAUSS = "B2"


def key(frame_type, p_space):
    return "I_aligned" if frame_type == "I" else f"P_{p_space}"


def split_train(train, gate0_videos):
    if train.split != "train":
        raise ValueError("C1 calibration is pixel-readable only on Train")
    eligible = sorted(set(train.rows) - set(gate0_videos))
    generator = np.random.default_rng(FIT_SEED + 1)
    calibration = {str(video): train.rows[video] for video in sorted(
        generator.choice(eligible, CALIBRATION_VIDEOS, replace=False)
    )}
    fitting = {k: v for k, v in train.rows.items() if k not in calibration}
    return replace(train, rows=fitting), replace(train, rows=calibration)


def sample_gradient(sample, backbone):
    with torch.enable_grad():
        y = sample.aligned.detach().requires_grad_(True)
        decoded, _ = backbone.decode_aligned(y, sample.context, checkpointed=True)
        gradient, = torch.autograd.grad(F.mse_loss(decoded, sample.rgb), y)
    res = [(key(sample.frame_type, ALIGNED), gradient)]
    if not sample.context.intra:
        res.append((key("P", NATIVE), native_gradient(gradient, backbone.alignment)))
    return {name: value.double().square().mean((0, 2, 3)) for name, value in res}


def channel_rows(y):
    return y.movedim(1, -1).reshape(-1, y.shape[1])[::CALIBRATION_STRIDE]


def sample_values(sample):
    signature = key(sample.frame_type, ALIGNED)
    values = {signature: channel_rows(sample.aligned)}
    if not sample.context.intra:
        values[key("P", NATIVE)] = channel_rows(sample.native)
    return values


def fit_samples(train, backbone, plan):
    for row in plan:
        clip = train.read(row["video"], row["start"], row["frames"])
        for sample in clean_samples(clip, range(len(clip)), backbone):
            sensitivity = row["sensitivity"] and sample.context.frame_id in (0, len(clip) - 1)
            yield sample, sensitivity


def append_stats(storage, data):
    for name, value in data.items():
        storage[name].append(value)


def fit_all(train, backbone, identity):
    if train.split != "train":
        raise ValueError("C1 channel fitting requires Train")
    plan = training_plan(train, FIT_SEED)
    values, gradients = ({name: [] for name in FIT_SPACES} for _ in range(2))
    for sample, sensitivity in fit_samples(train, backbone, plan):
        append_stats(values, sample_values(sample))
        if sensitivity:
            append_stats(gradients, sample_gradient(sample, backbone))
    fits = {
        name: fit_channels(torch.cat(values[name]), torch.stack(gradients[name]).mean(0).float())
        for name in FIT_SPACES
    }
    counts = {name: {"sampled_rows": sum(len(v) for v in values[name]),
                     "gradient_frames": len(gradients[name])} for name in FIT_SPACES}
    save_checkpoint(FITTED, {"identity": identity, "plan": plan, "counts": counts,
                             "fits": {name: vars(fit) for name, fit in fits.items()}})
    print(f"Train fits complete on {len(plan)} clips", flush=True)
    return fits, plan, counts


def calibration_plan(calibration):
    rng = np.random.default_rng(FIT_SEED + 2)
    return [{"video": video, "start": int(rng.integers(row["original_frames"] - 3)), "frames": 4,
             "scored_local_frames": [0, 3]} for video, row in calibration.rows.items()]


def static_weights(fits, p_space):
    weight = (fits["I_aligned"].group_energy() + fits[key("P", p_space)].group_energy()) / 2
    return tuple(int(round(x * 1_000_000)) for x in weight.tolist())


def scan_tables(fits):
    return {(space, n): scan_schedule(n, weights=static_weights(fits, space))
            for space, n in product((ALIGNED, NATIVE), SCHEDULES)}


def sparse_ordered(sample, space, fit):
    return grouped(fit.order_values(coordinate_value(sample.native, sample.aligned, space))[:, :TOP])


def sparse_latent(backbone, fit, space, observations, sigma=None):
    return aligned_value(backbone, fit.restore_rank(observations.ordered(sigma)), space)


@torch.no_grad()
def score_sample(sample, space, fit, backbone, fsq, tables, fields):
    values = sparse_ordered(sample, space, fit)
    scales = fit.scales[:4]
    for schedule, version in product(SCHEDULES, VERSIONS):
        slots = tables[schedule]
        packet = send(fsq, values, scales, slots, version)
        for budget in BUDGETS:
            part = receive(fsq, packet.payload[:, :budget], scales, slots[:budget], SHAPE, version)
            for sigma in (0.0, *SIGMAS):
                latent = sparse_latent(backbone, fit, space, part, sigma or None)
                reconstruction, _ = backbone.decode_aligned(latent, sample.context)
                mse = F.mse_loss(reconstruction, sample.rgb).item()
                yield {**fields, "coordinate": space, "schedule": schedule, "version": version,
                       "budget": budget, "method": GAUSS if sigma else ZERO, "sigma": sigma, "mse": mse}


def calibration_rows(calibration, codec, plan, fits, tables):
    fsq = FSQ().cuda()
    on_device = {(space, n): torch.tensor(table, device="cuda") for (space, n), table in tables.items()}
    for item in plan:
        video = item["video"]
        clip = calibration.read(video, item["start"], item["frames"])
        for hist, sample in history_samples(clip, item["scored_local_frames"], video, codec):
            fields = frame_fields(sample, video, hist, "train_calibration") | {
                "source_frame": item["start"] + sample.context.frame_id,
            }
            for policy in (ALIGNED, NATIVE):
                space = ALIGNED if sample.context.intra else policy
                fit = fits[key(sample.frame_type, space)]
                yield from score_sample(sample, space, fit, codec.backbone, fsq,
                                        {n: on_device[policy, n] for n in SCHEDULES},
                                        fields | {"p_policy": policy})
        print(f"Train calibration {video}", flush=True)


def case_rows(rows, space, schedule, sigma):
    """For I, score the same actual candidate table even when P is native."""
    return [r for r in rows if r["p_policy"] == space and r["schedule"] == schedule
            and r["sigma"] == sigma and r["version"] == "fsq"]


def closed_budget_gain(interpolated, zero):
    def pooled(rows):
        p = [r["mse"] for r in rows if r["frame_type"] == "P" and r["budget"] == 510 and r["history"] == LOW]
        return psnr(float(np.mean(p)))

    return pooled(interpolated) - pooled(zero)


def candidate_ranking(rows):
    ranked = []
    for space, name, sigma in product((ALIGNED, NATIVE), SCHEDULES, SIGMAS):
        case = case_rows(rows, space, name, sigma)
        slices = [[r["mse"] for r in case if (r["frame_type"], r["history"], r["budget"]) == (f, h, k)]
                  for f, h, k in product(("I", "P"), (CLEAN, LOW), BUDGETS)]
        score = float(np.mean([psnr(float(np.mean(chunk))) for chunk in slices]))
        gain = closed_budget_gain(case, case_rows(rows, space, name, 0))
        ranked.append({"p_coordinate": space, "schedule": name, "sigma": sigma,
                       "mean_slice_global_psnr": score, "matched_low_p510_gain_db": gain})
    return sorted(ranked, key=lambda r: -r["mean_slice_global_psnr"])


def calibrate(config, identity, codec):
    train = Catalog.open(config, "train")
    if identity["manifests"] != train.manifest_hashes:
        raise RuntimeError("Train catalog differs from the frozen source identity")
    fit_catalog, cal_catalog = split_train(train, config["calibration"]["videos"])
    fits, plan, counts = fit_all(fit_catalog, codec.backbone, identity)
    cal_plan = calibration_plan(cal_catalog)
    tables = scan_tables(fits)
    rows = list(calibration_rows(cal_catalog, codec, cal_plan, fits, tables))
    ranked = candidate_ranking(rows)
    winner = ranked[0]
    space, name = winner["p_coordinate"], winner["schedule"]
    write_rows("diag05/calibration", rows,
               ("p_policy", "coordinate", "history", "frame_type", "schedule", "version", "budget", "method", "sigma"),
               ranking=ranked, plans=cal_plan)
    locked = {
        "winner": winner, "slots": list(tables[space, name]), "shape": list(SHAPE), "identity": identity,
        "s3_static_weights": {p: static_weights(fits, p) for p in (ALIGNED, NATIVE)},
        "fit_sha256": file_hash(FITTED), "fit_plan": plan, "fit_counts": counts, "calibration_plan": cal_plan,
        "fit_statistics": {name: fit.record() for name, fit in fits.items()}, "locked_at": timestamp(),
        "selection": "highest mean of 20 I/P × history × budget global-MSE PSNR scores on FSQ B2",
        "phase_1_p510_gain_db": THRESHOLD_DB, "phase_2_dense_gain_db": THRESHOLD_DB,
        "phase_2_requires_lower_lpips_both_seeds": True,
    }
    write_json(LOCKED, locked)
    print(f"Locked on Train: {winner}", flush=True)
    return locked


def load_fits(artifact):
    return {name: ChannelFit(**data) for name, data in artifact["fits"].items()}
