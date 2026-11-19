"""No-learning matched and independent closed-loop probes on the sealed frame IDs."""

from dataclasses import replace
from itertools import product

import torch
from torch.nn import functional as F

from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.diagnostics.diag0.closed import same_state
from aip_vtok.diagnostics.diag0.io import Catalog, json_at, native_control
from aip_vtok.evaluation.fix import HORIZON_VIDEO, rms
from aip_vtok.evaluation.fix_metrics import entropy_bits
from aip_vtok.evaluation.gates import distortion_summary
from aip_vtok.runtime import file_hash, write_json
from aip_vtok.tokenizer.fsq import FSQ

from .artifacts import DIAG0, LOCKED, OUTPUT, THRESHOLD_DB, timestamp, write_rows
from .calibrate import GAUSS, ZERO, key, sparse_latent, sparse_ordered
from .channels import ALIGNED, CHANNELS, FILLS, NATIVE, PADDED, RANKS, aligned_value, coordinate_value
from .frames import capture, frame_fields, history_samples
from .metrics import FrameMetrics, metric_suite, quality_backslides
from .recovery_net import commit_reference
from .scan import BUDGETS, DENSE, REF_BUDGET, SHAPE, TOP
from .tokens import Packet, dense_roundtrip, fsq_counts, grouped, id_entropy, receive, send, wire_cost

HORIZON = 64
D0_OUT = OUTPUT / "diag05"
R0 = "R0-aligned"
SPARSE_METHODS = (ZERO, GAUSS)


def p_space(sample, lock):
    return ALIGNED if sample.context.intra else lock["winner"]["p_coordinate"]


def check_roundtrip(sample, backbone, fit, space, ordered):
    restored = fit.restore_rank(ordered)
    orig = coordinate_value(sample.native, sample.aligned, space)
    axis_max = (restored - orig).abs().max().item()
    aligned = aligned_value(backbone, restored, space)
    alignment_max = (aligned - sample.aligned).abs().max().item()
    native_max = 0.0
    if space == NATIVE:
        native_max = (backbone.alignment.restore(aligned, False) - sample.native).abs().max().item()
    if max(axis_max, alignment_max, native_max) >= 1e-5:
        raise RuntimeError(f"128D FP32 round trip exceeded tolerance at frame {sample.context.frame_id}")
    return {"coordinate_max_abs": axis_max, "aligned_max_abs": alignment_max, "native_max_abs": native_max}


def rank_sources(fsq, fit, sample, space):
    original = coordinate_value(sample.native, sample.aligned, space)
    ordered = fit.order_values(original)
    padded = F.pad(ordered, (0, 0, 0, 0, 0, PADDED - CHANNELS))
    quantized, packet = dense_roundtrip(fsq, grouped(padded), fit.scales)
    return ordered, quantized.flatten(1, 2)[:, :CHANNELS], packet


def decode_metrics(backbone, sample, scorer, aligned):
    rgb, _ = backbone.decode_aligned(aligned, sample.context)
    return scorer(rgb, aligned, sample.aligned)


def rank_curves(backbone, sample, scorer, fit, space, fields, sources):
    ordered, quantized, _ = sources
    check = check_roundtrip(sample, backbone, fit, space, ordered)
    labels = {**fields, "coordinate": space}
    for (version, values), rank in product((("cont", ordered), ("fsq", quantized)), RANKS):
        fills = ("zero_fill",) if rank == CHANNELS else FILLS
        for fill in fills:
            restored = fit.restore_rank(values[:, :rank], fill)
            aligned = aligned_value(backbone, restored, space)
            yield {**labels, "version": version, "rank": rank, "fill": fill,
                   **decode_metrics(backbone, sample, scorer, aligned), **(check if rank == CHANNELS else {})}


def fsq_health(counts, saturated):
    records = []
    for group, dims in enumerate(counts):
        levels = (8, 8, 8, 5, 5, 5)
        total = int(dims.sum())
        boundary = sum(dim[0] + dim[level - 1] for dim, level in zip(dims, levels))
        records.append({"group": group, "tokens": total // 6, "boundary_code_ratio": float(boundary / total),
                        "tanh_saturation_ratio": float(saturated[group] / total),
                        "dim_entropy_bits": [entropy_bits(dim[:level]) for dim, level in zip(dims, levels)]})
    return records


def dense_stats(fsq, packet):
    packet = Packet("fsq", packet.payload[:, :4], packet.saturated[:, :4])
    counts, saturated = fsq_counts(fsq, packet)
    return fsq_health(counts, saturated)


def rate(budget, received, pixels, ids):
    out = {}
    for prefix, n in (("display", budget), ("received", received)):
        ledger = wire_cost(n, id_entropy(ids[:, :n]))
        out.update({f"{prefix}_{name}": value for name, value in ledger.items()})
        for kind, name in (("raw", "raw_total_bytes"), ("entropy_est", "entropy_total_est_bytes")):
            out[f"{prefix}_bpp_{kind}"] = ledger[name] * 8 / pixels
    return out


def check_prefix(fsq, values, scales, slots, packet):
    for k in BUDGETS:
        separately = send(fsq, values, scales, slots[:k], "fsq").payload
        if not torch.equal(separately, packet.payload[:, :k]):
            raise RuntimeError(f"FSQ ID prefix changed at K={k}")


def matched_sparse(backbone, sample, scorer, fit, space, fsq, slots, sigma, fields):
    values = sparse_ordered(sample, space, fit)
    scales = fit.scales[:4]
    for version in ("cont", "fsq"):
        packet = send(fsq, values, scales, slots, version)
        if version == "fsq":
            check_prefix(fsq, values, scales, slots, packet)
        for k in BUDGETS:
            part = receive(fsq, packet.payload[:, :k], scales, slots[:k], SHAPE, version)
            ledger = rate(k, max(k, REF_BUDGET[sample.context.intra]), sample.rgb.shape[-2:].numel(),
                          packet.payload) if version == "fsq" else {}
            for method, width in ((ZERO, None), (GAUSS, sigma)):
                latent = sparse_latent(backbone, fit, space, part, width)
                yield {**fields, "coordinate": space, "budget": k, "method": method, "version": version,
                       "observed_groups": [int(n) for n in part.mask.sum((0, 2, 3)).tolist()],
                       **decode_metrics(backbone, sample, scorer, latent), **ledger}


class Matched:
    def __init__(self, codec, suite, fits, lock):
        self.codec, self.suite, self.fits, self.lock = codec, suite, fits, lock
        self.fsq = FSQ().cuda()
        self.slots = torch.tensor(lock["slots"], device="cuda")
        self.rank_rows, self.dense_rows, self.sparse_rows = [], [], []
        self.metric_error = {}

    def measure_frame(self, sample, fields):
        backbone = self.codec.backbone
        scorer = FrameMetrics(self.suite, sample.rgb)
        direct, _ = backbone.decode_aligned(sample.aligned, sample.context)
        measured = scorer(direct, sample.aligned, sample.aligned)
        if not self.metric_error:
            expected = self.suite(scorer.rgb, direct.to(memory_format=torch.channels_last))
            self.metric_error = {m: abs(expected[m] - measured[m]) for m in ("mse", "lpips", "dists")}
            if max(self.metric_error.values()) >= 3e-6:
                raise RuntimeError(f"native-resolution metric cache differs: {self.metric_error}")
        self.rank_rows.append({**fields, "coordinate": R0, "rank": 128, "fill": "zero_fill",
                               "version": "cont", **measured})
        selected = p_space(sample, self.lock)
        spaces = (ALIGNED,) if sample.context.intra else (ALIGNED, NATIVE)
        for space in spaces:
            self.measure_ranks(backbone, sample, scorer, fields, space, selected)
        self.sparse_rows.extend(matched_sparse(
            backbone, sample, scorer, self.fits[key(sample.frame_type, selected)], selected,
            self.fsq, self.slots, self.lock["winner"]["sigma"], fields,
        ))

    def measure_ranks(self, backbone, sample, scorer, fields, space, selected):
        fit = self.fits[key(sample.frame_type, space)]
        sources = rank_sources(self.fsq, fit, sample, space)
        curves = list(rank_curves(backbone, sample, scorer, fit, space, fields, sources))
        self.rank_rows.extend(curves)
        if space == selected:
            health = dense_stats(self.fsq, sources[2])
            for row in curves:
                if row["rank"] == TOP and row["fill"] == "zero_fill":
                    self.dense_rows.append(row | {"budget": DENSE, "method": f'B0-{row["version"]}',
                                                  "fsq": health if row["version"] == "fsq" else None})

    def restore(self, path):
        record = json_at(path)
        self.rank_rows.extend(record["rank"])
        self.dense_rows.extend(record["dense"])
        self.sparse_rows.extend(record["sparse"])
        self.metric_error = record["metric_cache_error"]

    def save(self, frames):
        details = {"scan_lock_sha256": file_hash(LOCKED), "source_frame_ids": frames,
                   "metric_cache_error": self.metric_error, "completed": True}
        write_rows("diag05/rank", self.rank_rows, ("history", "frame_type", "coordinate", "version", "fill", "rank"),
                   **details)
        write_rows("diag05/dense", self.dense_rows, ("history", "frame_type", "coordinate", "method"), **details)
        keys = ("history", "frame_type", "coordinate", "method", "version")
        write_rows("diag05/matched_sparse", self.sparse_rows, (*keys, "budget"), **details,
                   fsq_prefixes_verified=True, quality_backslides=quality_backslides(self.sparse_rows, keys))


@torch.no_grad()
def run_matched(validation, codec, suite, fits, lock):
    frames = json_at(DIAG0 / "provenance.json")["validation_frame_ids"]
    measured = Matched(codec, suite, fits, lock)
    for ordinal, (video, ids) in enumerate(frames.items()):
        partial = D0_OUT / f".matched_{ordinal}.json"
        if partial.is_file() and json_at(partial)["lock"] == file_hash(LOCKED):
            measured.restore(partial)
            continue
        rank_before, dense_before, sparse_before = map(len, (measured.rank_rows, measured.dense_rows,
                                                             measured.sparse_rows))
        clip = validation.read(video, 0, max(ids) + 1)
        for history, sample in history_samples(clip, ids, video, codec):
            measured.measure_frame(sample, frame_fields(sample, video, history, "validation"))
            print(f"Diag-0.5 {video} {history} frame {sample.context.frame_id}", flush=True)
        write_json(partial, {"lock": file_hash(LOCKED), "rank": measured.rank_rows[rank_before:],
                             "dense": measured.dense_rows[dense_before:], "sparse": measured.sparse_rows[sparse_before:],
                             "metric_cache_error": measured.metric_error})
    measured.save(frames)


def reference_details(backbone, sample, state, budget):
    _, feature = backbone.decode_native(sample.native, sample.context)
    return {
        "reference_budget": budget, "reference_latent_rms": rms(state.latent),
        "reference_to_native_rms": rms(state.latent) / max(rms(sample.aligned), 1e-12),
        "reference_latent_mse": F.mse_loss(state.latent, sample.aligned).item(),
        "projected_feature_rms": rms(state.feature),
        "reference_feature_l1": F.l1_loss(state.feature, feature).item(),
        "memory_rms": rms(state.memory), "state_sync": True,
    }


@torch.no_grad()
def sparse_rollout(clip, video, codec, suite, lock, fits, method, model=None):
    backbone, projector = codec.backbone, codec.projector
    sender, receiver = ReferenceState(), ReferenceState()
    slots = torch.tensor(lock["slots"], device="cuda")
    fsq = FSQ().cuda()
    for frame_id, image in enumerate(clip):
        sample = capture(backbone, sender, frame_id, image)
        rx = backbone.begin_frame(receiver, frame_id)
        space = p_space(sample, lock)
        fit = fits[key(sample.frame_type, space)]
        scales = fit.scales[:4]
        values = sparse_ordered(sample, space, fit)
        packet = send(fsq, values, scales, slots)
        sigma = None if method == ZERO else lock["winner"]["sigma"]
        ref = REF_BUDGET[rx.intra]

        def recover(k, context):
            part = receive(fsq, packet.payload[:, :k], scales, slots[:k], SHAPE, packet.version)
            latent = sparse_latent(backbone, fit, space, part, sigma)
            return latent if model is None else model(latent, part.mask, fit.indices[:TOP], context)

        sender_latent = recover(ref, sample.context)
        sender = commit_reference(sample.context, sender_latent, projector(sender_latent, rx.intra))
        rx_sample = replace(sample, context=rx)
        scorer = FrameMetrics(suite, sample.rgb)
        renders = {}
        for k in BUDGETS:
            if k <= ref:
                renders[k] = decode_metrics(backbone, rx_sample, scorer, recover(k, rx))
        receiver_latent = recover(ref, rx)
        receiver = commit_reference(rx, receiver_latent, projector(receiver_latent, rx.intra))
        if not same_state(sender, receiver):
            raise RuntimeError(f"sender/receiver detached states differ at frame {frame_id}")
        diagnostics = reference_details(backbone, sample, receiver, ref)
        for k in BUDGETS:
            if k > ref:
                renders[k] = decode_metrics(backbone, rx_sample, scorer, recover(k, rx))
            fields = frame_fields(sample, video, "independent_closed", "validation")
            yield {**fields, "method": method, "budget": k, "coordinate": space, "version": "fsq",
                   **renders[k], **diagnostics,
                   **rate(k, max(k, ref), sample.rgb.shape[-2:].numel(), packet.payload)}


def p510(rows, method):
    return distortion_summary([r for r in rows
                               if r["method"] == method and r["budget"] == 510 and r["frame_type"] == "P"])


def run_closed(validation, codec, suite, fits, lock):
    if json_at(D0_OUT / "matched_sparse.json")["scan_lock_sha256"] != file_hash(LOCKED):
        raise RuntimeError("Diag-0.5 closed loop requires complete matched data for this lock")
    clip = validation.read(HORIZON_VIDEO, 0, HORIZON)
    rows = []
    for method in SPARSE_METHODS:
        rows.extend(sparse_rollout(clip, HORIZON_VIDEO, codec, suite, lock, fits, method))
        print(f"Diag-0.5 closed {method} complete at {HORIZON} frames", flush=True)
    write_rows("diag05/closed", rows, ("method", "frame_type", "budget"),
               native_dmc_ld=native_control(), fsq_prefixes_verified_in_matched=True,
               quality_backslides=quality_backslides(rows, ("method", "frame_type")))
    calibration_gain = lock["winner"]["matched_low_p510_gain_db"]
    closed_gain = p510(rows, GAUSS)["psnr_global_mse"] - p510(rows, ZERO)["psnr_global_mse"]
    passed = min(calibration_gain, closed_gain) >= THRESHOLD_DB
    write_json(D0_OUT / "decision.json",
               {"completed": True, "passes_for_c1b": passed, "locked_sha256": file_hash(LOCKED),
                "calibration_matched_p510_gain_db": calibration_gain, "closed_p510_gain_db": closed_gain,
                "minimum_gain_db": THRESHOLD_DB, "new_closed_rows": len(rows), "closed_frames_per_method": HORIZON,
                "state_sync": all(r["state_sync"] for r in rows), "finished": timestamp()})


def evaluate(config, codec, fits, lock, stage):
    validation = Catalog.open(config, "validation")
    suite = metric_suite()
    if stage in ("matched", "all"):
        run_matched(validation, codec, suite, fits, lock)
    if stage in ("closed", "all"):
        run_closed(validation, codec, suite, fits, lock)
