"""D0-A through D0-D reuse each captured encoder and decoder context."""

from dataclasses import dataclass
from itertools import product

import torch
from torch.nn import functional as F

from aip_vtok.decoder.causal_state import FrameContext, ReferenceState
from aip_vtok.evaluation.fix import rgb_frame
from aip_vtok.evaluation.tokenizer_oracle import continuous_codes, unpack_patches
from aip_vtok.models.codec import EncodedFrame
from aip_vtok.tokenizer.fsq import FSQ
from aip_vtok.training.prefix import prefix_latents

from .basis import fp32_channel_matmul, latent_grid, latent_rows
from .fit import NATIVE_ORDER, ORDERS, RANKS
from .io import gpu_snapshot, write_probe
from .metrics import Measurements, measure
from .packets import FREE, LAYOUTS, VERSIONS, Transform
from .sparse import BUDGETS, REF_BUDGET

CLEAN = "r0_aligned"
LOW = "fix_a_low_reference"
CHANNEL_CONTROLS = (6, 12, 24, 48)


def validation_frames(length):
    return (0, 1, 8, 16, 31, 32, 33, 40, 48, length - 1)


@dataclass
class Frame:
    video: str
    history: str
    context: FrameContext
    rgb: torch.Tensor
    y: torch.Tensor
    direct: torch.Tensor
    encoded: EncodedFrame | None = None

    def fields(self):
        return {
            "video": self.video,
            "frame": self.context.frame_id,
            "frame_type": "I" if self.context.intra else "P",
            "history": self.history,
            "matched_six": self.context.frame_id < 2,
            "split": "validation",
        }


@torch.no_grad()
def clean_frames(clip, selected, video, backbone):
    state = ReferenceState()
    for index, image in enumerate(clip):
        rgb = rgb_frame(image)
        context = backbone.begin_frame(state, index)
        y = backbone.aligned_latent(rgb, context)
        reconstruction, feature = backbone.decode_aligned(y, context)
        state = context.commit_reference_once(feature)
        if index in selected:
            yield Frame(video, CLEAN, context, rgb, y, reconstruction)


@torch.no_grad()
def low_frames(clip, selected, video, codec):
    state = ReferenceState()
    for index, image in enumerate(clip):
        rgb = rgb_frame(image)
        encoded = codec.encode_frame(rgb, state, index)
        context, y = encoded.context, encoded.native_latent
        state = codec.prefix_session(encoded, state, REF_BUDGET[context.intra]).commit_reference_once()
        if index in selected:
            reconstruction, _ = codec.backbone.decode_aligned(y, context)
            yield Frame(video, LOW, context, rgb, y, reconstruction, encoded)


def rank_coordinates(fit, y, order):
    basis, indices = fit.basis, fit.indices(order)
    if order == NATIVE_ORDER:
        vectors = torch.eye(len(indices), device=y.device)[:, indices]
        variances = basis.diagonal
    else:
        vectors, variances = basis.vectors[:, indices], basis.eigenvalues
    z = fp32_channel_matmul(latent_rows(y) - basis.mean, vectors)
    return z, vectors, variances, indices.cpu()


def full_rank_latent(fit, y, order):
    z, vectors, _, _ = rank_coordinates(fit, y, order)
    return latent_grid(fp32_channel_matmul(z, vectors.T) + fit.basis.mean, y.shape[-2:])


def rank_latents(fit, y, order):
    basis = fit.basis
    z, vectors, variances, indices = rank_coordinates(fit, y, order)
    for rank in RANKS[:-1]:
        rows = basis.restore(z[:, :rank], vectors[:, :rank])
        retained = float(variances[indices[:rank]].sum() / variances.sum())
        yield rank, latent_grid(rows, y.shape[-2:]), retained


def block_method(fit, frame, block):
    y = frame.y
    shape = y.shape[-2:]
    target = F.interpolate(y, size=tuple(size // block for size in shape), mode="bilinear", align_corners=False)
    yield "bilinear", 128, F.interpolate(target, size=shape, mode="bilinear", align_corners=False)
    basis = fit.block2 if block == 2 else fit.block4
    for rank in (6, 128):
        yield "block_pca", rank, basis.project(y, rank, block)


def historical_methods(codec, frame, inverse):
    tokenizer, encoded = codec.tokenizer, frame.encoded
    plan = encoded.tokens.plan
    positions = plan.positions[plan.pass_ids == 0]
    centered = frame.y - encoded.prediction
    coeffs = continuous_codes(tokenizer.lifts[0], unpack_patches(centered, 4)[:, positions], inverse)
    learned = tokenizer.synthesize_pass(coeffs, positions, 0, plan.latent_shape) + encoded.prediction
    yield "fix_a_lift_oracle", 4, 6, learned
    for budget, name in ((510, "fix_a_pass0"), (8192, "fix_a_final")):
        _, y = prefix_latents(codec, encoded.tokens, encoded.prediction, budget, differentiable=False)
        yield name, 4 if budget == 510 else 1, 6, y


class MatchedProbes:
    def __init__(self, codec, suite, fits):
        self.codec, self.suite, self.fits = codec, suite, fits
        self.fsq = FSQ().cuda()
        lift = codec.tokenizer.lifts[0].weight
        self.inverse = torch.linalg.pinv(lift.cpu().double()).to(lift)
        shared = ("history", "frame_type")
        self.a = Measurements((*shared, "order", "rank"))
        self.b = Measurements((*shared, "method", "block", "rank"))
        self.c = Measurements((*shared, "layout", "order", "version", "budget"))
        self.d = Measurements(self.c.keys)

    def measure(self, frame, y):
        rec, _ = self.codec.backbone.decode_aligned(y, frame.context)
        return measure(self.suite, frame.rgb, rec, y, frame.y)

    def ranks(self, frame, fields, fit, compare_blocks):
        direct = measure(self.suite, frame.rgb, frame.direct, frame.y, frame.y)
        self.a.add({**fields, "order": "r0", "rank": 128, **direct})
        for order in (*ORDERS, NATIVE_ORDER):
            y = full_rank_latent(fit, frame.y, order)
            measured = self.measure(frame, y)
            maximum = (y - frame.y).abs().max().item()
            delta = measured["psnr"] - direct["psnr"]
            if maximum >= 1e-5 or abs(delta) >= 1e-4:
                raise RuntimeError(f"128D FP32 self-check failed: {fields}, {order}, {maximum}, {delta}")
            self.a.add({
                **fields, "order": order, "rank": 128, "variance_explained_train": 1.0, **measured,
                "full_rank_max_abs": maximum, "full_rank_psnr_delta": delta,
            })
        for order in (*ORDERS, NATIVE_ORDER):
            for rank, y, retained in rank_latents(fit, frame.y, order):
                measured = self.measure(frame, y)
                row = {**fields, "order": order, "rank": rank, "variance_explained_train": retained, **measured}
                self.a.add(row)
                if compare_blocks and order == ORDERS[0] and rank in CHANNEL_CONTROLS:
                    self.record_block(frame, fields, "channel_pca", 1, rank, measured)
        if compare_blocks:
            self.record_block(frame, fields, "r0", 1, 128, direct)

    def record_block(self, frame, fields, method, block, rank, measured, basis=None):
        energy = frame.y.square().mean().item()
        row = {
            **fields, "method": method, "block": block, "rank": rank, **measured,
            "latent_energy_zero": energy, "residual_fraction_zero": measured["latent_mse"] / energy,
        }
        if basis is not None:
            centered = (latent_rows(frame.y, block) - basis.mean).square().mean().item()
            row.update(
                residual_fraction_train_centered=measured["latent_mse"] / centered,
                latent_energy_train_centered=centered,
                train_optimal_centered_residual=1 - basis.explained(rank),
            )
        self.b.add(row)

    def blocks(self, frame, fields, fit):
        for block in (2, 4):
            for method, rank, y in block_method(fit, frame, block):
                basis = {2: fit.block2, 4: fit.block4}[block] if method == "block_pca" else None
                self.record_block(frame, fields, method, block, rank, self.measure(frame, y), basis)
        if frame.encoded is not None:
            for method, block, rank, y in historical_methods(self.codec, frame, self.inverse):
                self.record_block(frame, fields, method, block, rank, self.measure(frame, y))

    def sparse(self, frame, fields, fit):
        for layout, order, version in product((FREE, *LAYOUTS), ORDERS, VERSIONS):
            transform = Transform(fit, order, version, layout, self.fsq, frame.y.shape[-2:])
            target = self.c if layout == FREE else self.d
            for budget, packet in transform.encode(frame.y, BUDGETS).items():
                y = transform.receive(packet)
                ledger, counts, saturated = transform.ledger(packet, budget, frame.rgb.shape[-2:].numel())
                row = {
                    **fields, "layout": layout, "order": order, "version": version, "budget": budget,
                    **self.measure(frame, y), **ledger, "real_bitstream": False,
                }
                if counts is not None:
                    row["tanh_saturation_ratio"] = int(saturated.sum()) / int(counts.sum())
                target.add(row, counts, saturated)

    @torch.no_grad()
    def evaluate(self, frame):
        fields = frame.fields()
        fit = self.fits[frame.context.intra]
        shared = frame.history == CLEAN or fields["matched_six"]
        self.ranks(frame, fields, fit, shared)
        if shared:
            self.blocks(frame, fields, fit)
            self.sparse(frame, fields, fit)

    def save(self):
        for probe, measurements in zip("ABCD", (self.a, self.b, self.c, self.d)):
            write_probe(probe, measurements.rows, measurements.aggregate())


def run_matched(catalog, codec, suite, fits, videos):
    probes = MatchedProbes(codec, suite, fits)
    frame_manifest = {}
    for video in videos:
        length = min(64, catalog.rows[video]["original_frames"])
        selected = validation_frames(length)
        frame_manifest[video] = list(selected)
        clip = catalog.read(video, 0, length)
        # no_grad must cover iteration over the generator as well as its body.
        with torch.no_grad():
            for frames in (clean_frames(clip, selected, video, codec.backbone), low_frames(clip, selected, video, codec)):
                gpu_snapshot()
                for index, frame in enumerate(frames, 1):
                    probes.evaluate(frame)
                    print(f"A-D {frame.history} {video} frame={frame.context.frame_id} ({index}/{len(selected)})", flush=True)
        probes.save()
    return frame_manifest
