"""Train-only streamed moments, frozen-decoder gradients and FSQ scale calibration."""

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.evaluation.fix import rgb_frame
from aip_vtok.runtime import save_checkpoint
from aip_vtok.tokenizer.fsq import FSQ

from .basis import Basis, fit_basis, latent_grid, latent_rows
from .sparse import GROUPS, decode_fsq, encode_fsq, fit_scale

FIT_SEED = 7300
TRAIN_WINDOW = 4
SENSITIVITY_WINDOWS = 12
CALIBRATION_STRIDE = 16
ORDERS = ("pca_variance", "pca_rgb_sensitivity")
NATIVE_ORDER = "native_rgb_sensitivity"
RANKS = (6, 12, 18, 24, 32, 48, 64, 96, 128)


@dataclass(frozen=True)
class Fit:
    basis: Basis
    block2: Basis
    block4: Basis
    sensitivity: torch.Tensor
    native_sensitivity: torch.Tensor
    anchor_scale: torch.Tensor
    scales: dict
    residual_scales: dict

    def indices(self, order):
        if order == "pca_variance":
            return torch.arange(128, device=self.basis.mean.device)
        scores = self.native_sensitivity if order == NATIVE_ORDER else self.sensitivity
        return torch.argsort(scores, descending=True, stable=True)

    def axes(self, order):
        return self.basis.vectors[:, self.indices(order)]

    def weights(self, order):
        if order == "pca_variance":
            return 1.0
        sensitive = self.sensitivity[self.indices(order)[:GROUPS * 6]]
        return (sensitive / sensitive.mean().clamp_min(1e-30)).reshape(GROUPS, 6).float()


def training_plan(catalog, seed):
    rng = np.random.default_rng(seed)
    videos = sorted(catalog.rows)
    selected = rng.choice(len(videos), size=min(SENSITIVITY_WINDOWS, len(videos)), replace=False)
    return [
        {
            "video": video, "start": int(rng.integers(catalog.rows[video]["original_frames"] - TRAIN_WINDOW + 1)),
            "frames": TRAIN_WINDOW, "sensitivity": index in selected,
        }
        for index, video in enumerate(videos)
    ]


def distortion_gradient(rgb, y, context, backbone):
    with torch.enable_grad():
        y = y.detach().requires_grad_(True)
        image, _ = backbone.decode_aligned(y, context)
        gradient, = torch.autograd.grad(F.mse_loss(image, rgb), y)
    return latent_rows(gradient).double()


@torch.no_grad()
def training_window(catalog, backbone, plan):
    clip = catalog.read(plan["video"], plan["start"], plan["frames"])
    state, values = ReferenceState(), []
    for frame, image in enumerate(clip):
        context = backbone.begin_frame(state, frame)
        rgb = rgb_frame(image)
        y = backbone.aligned_latent(rgb, context)
        gradient = None
        if plan["sensitivity"] and frame in (0, TRAIN_WINDOW - 1):
            gradient = distortion_gradient(rgb, y, context, backbone)
        values.append((context.intra, y, gradient))
        if frame + 1 < len(clip):
            _, feature = backbone.decode_aligned(y, context)
            state = context.commit_reference_once(feature)
    return values


def train_latents(catalog, backbone, seed):
    if catalog.split != "train":
        raise ValueError("statistics require the train split before reading any frames")
    plan = training_plan(catalog, seed)
    return plan, (training_window(catalog, backbone, window) for window in plan)


def sensitivity_stats(gradients, basis):
    pooled = torch.cat(gradients)
    return (pooled @ basis.vectors.double()).square().mean(0), pooled.square().mean(0)


def anchor_scales(y, basis, fsq):
    axes = basis.vectors[:, :6]
    anchor = [basis.rotate(latent_rows(frame, 4), axes) for frame in y]
    scale = fit_scale(torch.cat(anchor))
    residuals = []
    for frame, codes in zip(y, anchor):
        ids, _ = encode_fsq(fsq, codes, scale)
        rows = basis.restore(decode_fsq(fsq, ids, scale), axes)
        residuals.append(frame - latent_grid(rows, frame.shape[-2:], 4))
    return scale, residuals


def group_scales(y, basis, orders, centered):
    samples = torch.cat([latent_rows(frame)[::CALIBRATION_STRIDE] for frame in y])
    rotated = basis.rotate(samples, centered=centered)
    return {key: fit_scale(rotated[:, index[:126]]).reshape(GROUPS, 6) for key, index in orders.items()}


def fit_frame_type(y, gradients, fsq):
    basis = fit_basis(y, 128)
    blocks = [fit_basis(y, 128, block) for block in (2, 4)]
    sensitivity, native = sensitivity_stats(gradients, basis)
    scale, residuals = anchor_scales(y, blocks[1], fsq)
    orders = dict(zip(ORDERS, (
        torch.arange(128, device=basis.mean.device), torch.argsort(sensitivity, descending=True, stable=True),
    )))
    return Fit(
        basis, *blocks, sensitivity, native, scale,
        group_scales(y, basis, orders, True), group_scales(residuals, basis, orders, False),
    )


def fit_probes(catalog, backbone):
    plan, windows = train_latents(catalog, backbone, FIT_SEED)
    latents, gradients = {True: [], False: []}, {True: [], False: []}
    for index, window in enumerate(windows):
        for intra, y, grad in window:
            latents[intra].append(y)
            if grad is not None:
                gradients[intra].append(grad)
        if (index + 1) % 8 == 0 or index + 1 == len(plan):
            print(f"fit train windows={index + 1}/{len(plan)}", flush=True)
    fsq = FSQ().to(latents[True][0].device)
    return {intra: fit_frame_type(latents[intra], gradients[intra], fsq) for intra in (True, False)}, plan


def save_fits(path, fits, plan, identity):
    payload = {}
    for name, fit in fits.items():
        payload[name] = {**vars(fit), **{key: vars(getattr(fit, key)) for key in ("basis", "block2", "block4")}}
    save_checkpoint(path, {"fits": payload, "train_plan": plan, "identity": identity})


def load_fits(path, identity):
    artifact = torch.load(path, map_location="cuda", weights_only=True)
    if artifact["identity"] != identity:
        raise ValueError("fitting inputs changed; rerun the train-only stage")
    for values in artifact["fits"].values():
        for key in ("basis", "block2", "block4"):
            values[key] = Basis(**values[key])
    return {intra: Fit(**fit) for intra, fit in artifact["fits"].items()}, artifact["train_plan"]
