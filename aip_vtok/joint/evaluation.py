"""Native full-frame rate, distortion and latency measured from transmitted bytes."""

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from aip_vtok.c1.frames import capture
from aip_vtok.c1.metrics import FrameMetrics, metric_suite
from aip_vtok.c1.scan import BUDGETS as LEGACY_BUDGETS
from aip_vtok.diagnostics.diag0.io import json_at
from aip_vtok.evaluation.fix import HORIZON_VIDEO, rms
from aip_vtok.evaluation.gates import psnr
from aip_vtok.runtime import write_json
from aip_vtok.sparse_decoder.models import SLD, DecoderReceiver
from aip_vtok.sparse_decoder.protocol import load_w1
from aip_vtok.sparse_rfsq.calibration import load_inputs, load_scan
from aip_vtok.sparse_rfsq.receiver import Receiver
from aip_vtok.sparse_rfsq.rollout import CausalPair
from cp_adaptok.metrics import summarize_latency

from .bitstream import FRAME_HEADER, GOP, PACKET_HEADER, PIXELS
from .checkpoint import CheckpointStore, load_model
from .codec import DecodeSession, timed
from .protocol import CHECKPOINTS, OUTPUT, SLD_500
from .scan import BUDGETS, REF_BUDGET
from .trainer import History

CLOSED, FIXED = "CLOSED", "FIXED"
JOINT = "Joint V1"
BASELINES = ("B2", "W1", "SLD-500")
LATE_START = 48
VIZ_FRAME = 55
VIZ_BUDGETS = (510, 2048, 8192, 32640, 65280)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def reference_fields(tx, rx) -> dict:
    row = {"reference_latent_rms": rms(rx.latent), "reference_feature_rms": rms(rx.feature), "state_sync": True}
    if rx.frame_id % GOP in (0, GOP - 1) or rx.frame_id == VIZ_FRAME:
        row.update(tx_reference_sha256=tx.fingerprint(), rx_reference_sha256=rx.fingerprint())
    return row


def transmitted_tokens(display: int, intra: bool) -> int:
    return max(display, REF_BUDGET[intra])


def read_packets(session: DecodeSession, frame, budget: int) -> float:
    """Append only newly transmitted packets, measuring parser and independent rANS work."""
    start_time = perf_counter()
    offset = FRAME_HEADER.size if session.received == 0 else frame.ends[session.received][0]
    end = frame.ends[budget][0]
    while offset < end:
        _, _, size, _ = PACKET_HEADER.unpack_from(frame.data, offset)
        next_offset = offset + PACKET_HEADER.size + size
        if next_offset > end:
            raise ValueError("packet crosses the declared byte-prefix boundary")
        session.append_packet(frame.data[offset:next_offset])
        offset = next_offset
    if session.received != budget:
        raise RuntimeError("received slot count differs from the selected wire prefix")
    return (perf_counter() - start_time) * 1000


def save_crop(directory, method, budget, rgb):
    from PIL import Image

    directory.mkdir(parents=True, exist_ok=True)
    crop = rgb[0, :, 432:648, 768:1152].permute(1, 2, 0)
    Image.fromarray(crop.mul(255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()).save(
        directory / f"{method}_{budget}.png"
    )


def save_stream(directory, label, encoded):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{label}.aipj").write_bytes(encoded.wire.data)
    np.save(directory / f"{label}.npy", encoded.ordered_ids.cpu().numpy().astype("<u2"), allow_pickle=False)


@torch.no_grad()
def joint_rows(codec, profile, history, rgb, frame_id, fields, stats, artifact_dir=None, crop_dir=None):
    torch.cuda.reset_peak_memory_stats(codec.device)
    frame = codec.encode_frame(rgb, history.tx, frame_id, profile)
    intra = frame.context.intra
    header = frame.wire.data[:FRAME_HEADER.size]
    tx = DecodeSession(codec, frame.context, profile)
    rx, ctx_ms = timed(codec.device, codec.begin_frame, history.rx, header, profile)
    ref = REF_BUDGET[intra]
    tx_decode_ms = read_packets(tx, frame.wire, ref)
    tx_state, tx_commit_ms = timed(codec.device, tx.commit_reference)
    rx_decode_ms = read_packets(rx, frame.wire, ref)
    rx_state, rx_commit_ms = timed(codec.device, rx.commit_reference)
    # The source is never supplied to either endpoint's reference recovery.
    history.submit(tx_state, rx_state)
    reference = reference_fields(tx_state, rx_state)
    host_ids = frame.ordered_ids.flatten().cpu().tolist()
    proxy = codec.entropy.bits(frame.ordered_ids, codec.slot_groups[profile.table], intra).cumsum(-1)[0].tolist()
    if artifact_dir is not None:
        save_stream(artifact_dir, f"{frame_id:04d}", frame)
    common = fields | reference | frame.latency | {
        "method": JOINT, "frame_type": "I" if intra else "P", "reference_budget": ref,
        "rx_context_ms": ctx_ms, "tx_reference_ms": tx_decode_ms + tx_commit_ms, "rx_reference_ms": rx_commit_ms,
    }
    tx_ms = sum(frame.latency.values()) + tx_decode_ms + tx_commit_ms
    for k in BUDGETS:
        sent = transmitted_tokens(k, intra)
        if sent > rx.received:
            rx_decode_ms += read_packets(rx, frame.wire, sent)
        if rx.ids != host_ids[:sent]:
            raise RuntimeError("receiver integer prefix differs from the encoded ID prefix")
        peak_before_metrics = torch.cuda.max_memory_allocated(codec.device) / 2**20
        latent, sld_ms = timed(codec.device, rx.latent, k)
        (rgb_rec, _), rgb_ms = timed(codec.device, codec.backbone.decode_aligned, latent, rx.context)
        peak = max(peak_before_metrics, torch.cuda.max_memory_allocated(codec.device) / 2**20)
        torch.cuda.reset_peak_memory_stats(codec.device)
        entropy_ms = frame.wire.ends[sent][2]
        decoder_ms = ctx_ms + rx_decode_ms + rx_commit_ms + sld_ms + rgb_ms
        metrics = stats(rgb_rec, latent, frame.aligned) if stats is not None else {}
        if crop_dir is not None and k in VIZ_BUDGETS:
            save_crop(crop_dir, JOINT, k, rgb_rec)
        rate = frame.wire.ledger(sent)
        rate["display_tokens"] = k
        yield common | rate | metrics | {
            "budget": k, "entropy_proxy_bits": proxy[sent - 1], "rate_basis": "actual_rans_plus_headers",
            "entropy_encode_ms": entropy_ms, "entropy_decode_ms": rx_decode_ms, "sld_ms": sld_ms,
            "rgb_decode_ms": rgb_ms, "total_encode_ms": tx_ms + entropy_ms, "total_decode_ms": decoder_ms,
            "end_to_end_ms": tx_ms + entropy_ms + decoder_ms, "peak_gpu_mib": peak,
            "prefix_ids_match": True, "token_bits": 16,
        }


def load_baselines(run, compare):
    lock = load_scan()
    inputs = load_inputs(run.legacy_config, run.legacy_identity)
    methods: dict[str, Receiver | DecoderReceiver] = {"B2": Receiver(lock["sigma"])}
    if compare:
        w1, _, _ = load_w1(lock, run.legacy_config)
        sld = SLD().to(device=run.codec.device, memory_format=torch.channels_last).eval().requires_grad_(False)
        sld.load_state_dict(torch.load(SLD_500, weights_only=True, map_location="cpu")["model"])
        methods.update({"W1": Receiver(lock["sigma"], w1), "SLD-500": DecoderReceiver(lock["sigma"], sld)})
    return inputs.legacy, inputs.cases("SP-1", lock["keys"]["SP-1"]), methods


def legacy_advance(codec, pair, raw, frame_id, cases, receiver):
    dev = raw.device
    sample, encoder_ms = timed(dev, capture, codec.backbone, pair.tx, frame_id, raw)
    case = cases[sample.frame_type]
    packet, fsq_scan_ms = timed(dev, case.encode, sample)
    context, context_ms = timed(dev, codec.backbone.begin_frame, pair.rx, frame_id)
    prefix = packet[:, :REF_BUDGET[context.intra]]

    def commit():
        tx = receiver(case, codec.backbone, sample.context, prefix)
        rx = receiver(case, codec.backbone, context, prefix)
        pair.commit(sample.context, tx, codec.projector(tx, context.intra),
                    context, rx, codec.projector(rx, context.intra))

    _, ref_ms = timed(dev, commit)
    times = {"encoder_ms": encoder_ms, "fsq_scan_ms": fsq_scan_ms, "rx_context_ms": context_ms,
             "paired_reference_ms": ref_ms}
    return sample, context, case, packet, times


def legacy_rows(codec, data, receiver, fields, stats, reference, crop_dir):
    sample, context, case, packet, times = data
    for k in LEGACY_BUDGETS:
        latent, sld_ms = timed(sample.rgb.device, receiver, case, codec.backbone, context, packet[:, :k])
        (rec, _), rgb_ms = timed(sample.rgb.device, codec.backbone.decode_aligned, latent, context)
        metrics = stats(rec, latent, sample.aligned) if stats is not None else {}
        if crop_dir is not None and k in VIZ_BUDGETS:
            save_crop(crop_dir, fields["method"], k, rec)
        sent = transmitted_tokens(k, context.intra)
        rate = case.rates[sent]
        encode_ms = times["encoder_ms"] + times["fsq_scan_ms"]
        decode_ms = times["rx_context_ms"] + sld_ms + rgb_ms
        yield fields | reference | times | metrics | {
            "frame_type": sample.frame_type, "budget": k, "display_tokens": k, "transmitted_tokens": sent,
            "reference_budget": REF_BUDGET[context.intra], "prefix_ids_match": True,
            "rate_basis": "legacy_raw_or_estimated", "token_bits": 16, "raw_token_bits": rate["payload_bits"],
            "raw_bpp": rate["payload_bits"] / PIXELS, "legacy_header_bytes": rate["header_bytes"],
            "legacy_estimated_entropy_bpp": rate["entropy_payload_bits"] / PIXELS,
            "sld_ms": sld_ms, "rgb_decode_ms": rgb_ms, "total_encode_ms": encode_ms,
            "total_decode_ms": decode_ms, "end_to_end_ms": encode_ms + decode_ms + times["paired_reference_ms"],
        }


def cohort(run, split, extend=False):
    rows = run.data.catalog(split)
    if split == "test":
        return {video: row["original_frames"] for video, row in sorted(rows.items())}
    selected = run.legacy_config["stage1"]["validation"]
    limit = selected["max_frames_per_video"]
    counts = {video: min(limit, rows[video]["original_frames"]) for video in selected["videos"]}
    if sum(counts.values()) != selected["total_frames"] or counts[HORIZON_VIDEO] != 64:
        raise RuntimeError("Validation differs from the original 174-frame cohort and 64-frame horizon")
    if extend:
        counts = {video: rows[video]["original_frames"] for video in counts}
    return counts


@torch.no_grad()
def evaluate(run, split, directory, *, fixed=False, compare=False, metrics=True, archive=False,
             extend=False):
    begun = perf_counter()
    codec = run.codec.eval()
    profile = codec.profile()
    suite = metric_suite() if metrics else None
    legacy, cases, receivers = load_baselines(run, compare) if fixed or compare else (None, None, {})
    rows = []
    for video, count in cohort(run, split, extend).items():
        closed = History.at(0)
        fixed_pair = CausalPair()
        baseline = {method: CausalPair() for method in receivers} if compare else {}
        clip = run.data.read(split, video, 0, count)
        for frame_id, raw in enumerate(clip):
            rgb = raw[None].to(codec.device, dtype=torch.float32).div_(255)
            stats = FrameMetrics(suite, rgb) if suite is not None else None
            fields = {"split": split, "video": video, "frame": frame_id, "gop_position": frame_id % GOP,
                      "standard_cohort": split == "test" or frame_id < 64,
                      "input_rgb_sha256": run.data.catalog(split)[video]["frame_hashes"][frame_id]["sha256"]}
            viz = (directory / "crops" if archive and split == "validation" and video == HORIZON_VIDEO
                   and frame_id == VIZ_FRAME else None)
            streams = directory / "streams" / video.replace(":", "_") if archive else None
            if viz is not None:
                for k in VIZ_BUDGETS:
                    save_crop(viz, "Original", k, rgb)
            rows.extend(joint_rows(codec, profile, closed, rgb, frame_id, fields | {"history": CLOSED},
                                   stats, streams / CLOSED if streams is not None else None, viz))
            if fixed:
                prior = History(fixed_pair.tx, fixed_pair.rx)
                fixed_data = legacy_advance(run.legacy_codec, fixed_pair, raw, frame_id, legacy, receivers["B2"])
                rows.extend(joint_rows(codec, profile, prior, rgb, frame_id, fields | {"history": FIXED},
                                       stats, streams / FIXED if streams is not None else None))
            for method, pair in baseline.items():
                receiver = receivers[method]
                closed_data = legacy_advance(run.legacy_codec, pair, raw, frame_id, cases, receiver)
                rows.extend(legacy_rows(codec, closed_data, receiver, fields | {"history": CLOSED, "method": method},
                                        stats, reference_fields(pair.tx, pair.rx), viz))
                if fixed:
                    sample, context, _, _, times = fixed_data
                    case = cases[sample.frame_type]
                    packet, fsq_ms = timed(codec.device, case.encode, sample)
                    times = times | {"fsq_scan_ms": fsq_ms}
                    rows.extend(legacy_rows(codec, (sample, context, case, packet, times), receiver,
                                            fields | {"history": FIXED, "method": method}, stats,
                                            reference_fields(fixed_pair.tx, fixed_pair.rx), None))
        print(f"{split}: {video} {count} native frames", flush=True)
    validate_rows(rows, cohort(run, split, extend), fixed, compare)
    run.verify_integrity()
    export_evaluation(rows, directory, split, profile, run.data.counts, perf_counter() - begun)
    return rows


def validate_rows(rows, counts, fixed, compare):
    expected_frames = sum(counts.values())
    expected = expected_frames * (len(BUDGETS) + (len(BASELINES) * len(LEGACY_BUDGETS) if compare else 0))
    expected *= 2 if fixed else 1
    keys = {(r["method"], r["history"], r["budget"], r["video"], r["frame"]) for r in rows}
    if len(keys) != expected or len(rows) != expected:
        raise RuntimeError("missing or duplicate per-frame evaluation rows")
    for row in rows:
        if not row["state_sync"] or not row["prefix_ids_match"]:
            raise RuntimeError("reference or prefix synchronization failed")
        if any(not math.isfinite(v) for v in row.values() if isinstance(v, float)):
            raise RuntimeError("nonfinite evaluation metric")


def aggregate(rows):
    by_point = defaultdict(list)
    for row in rows:
        by_point[row["method"], row["history"], row["budget"]].append(row)
    output = []
    for (method, history, k), points in by_point.items():
        first = points[0]
        late = [r for r in points if r["video"] == HORIZON_VIDEO and LATE_START <= r["frame"] < 64]
        summary = {"method": method, "history": history, "budget": k, "frames": len(points),
                   "status": "MEASURED", "rate_basis": first["rate_basis"], "token_bits": first["token_bits"],
                   **{key: mean(p[key] for p in points) for key in (
                       "mse", "psnr", "lpips", "dists", "latent_mse", "raw_bpp", "actual_bpp",
                       "entropy_bpp", "overhead_bpp", "legacy_estimated_entropy_bpp", "reference_latent_rms",
                   ) if key in first}}
        if "mse" in first:
            summary["global_psnr"] = psnr(summary["mse"])
            summary["mean_frame_psnr"] = summary.pop("psnr")
            if late:
                summary |= {"late_global_psnr": psnr(mean(p["mse"] for p in late)),
                            "late_lpips": mean(p["lpips"] for p in late),
                            "late_dists": mean(p["dists"] for p in late)}
        summary |= {f"{key}_{p}": value for key in ("total_encode_ms", "total_decode_ms", "end_to_end_ms")
                    for p, value in summarize_latency([r[key] for r in points]).items()}
        if "total_bytes" in first:
            summary["total_bytes"] = sum(p["total_bytes"] for p in points)
            summary["peak_gpu_mib"] = max(p["peak_gpu_mib"] for p in points)
        output.append(summary)
    for method, history in {(method, history) for method, history, _ in by_point if method in BASELINES}:
        for k in sorted(set(BUDGETS) - set(LEGACY_BUDGETS)):
            output.append({"method": method, "history": history, "budget": k, "status": "N/A"})
    return sorted(output, key=lambda r: (r["method"], r["history"], r["budget"]))


def export_evaluation(rows, directory, split, profile, reads, duration):
    rates = ("display_tokens", "transmitted_tokens", "token_bits", "raw_token_bits", "entropy_proxy_bits",
             "payload_bytes", "rans_state_bytes", "frame_header_bytes", "header_bytes", "packet_count",
             "total_bytes", "actual_bpp", "raw_bpp", "entropy_bpp", "overhead_bpp", "rate_basis",
             "legacy_header_bytes", "legacy_estimated_entropy_bpp")
    identifiers = ("split", "method", "history", "video", "frame", "frame_type", "budget")
    write_csv(directory / f"{split}_per_frame.csv", rows)
    write_csv(directory / f"{split}_metrics.csv", aggregate([r for r in rows if r["standard_cohort"]]))
    write_csv(directory / "rate_ledger.csv", [{k: r[k] for k in (*identifiers, *rates) if k in r} for r in rows])
    write_csv(directory / "latency_breakdown.csv", [
        {k: v for k, v in r.items() if k in identifiers or k.endswith("_ms") or k == "peak_gpu_mib"} for r in rows])
    write_csv(directory / "closed_loop_metrics.csv", [
        {k: v for k, v in r.items() if k in identifiers or "reference" in k or k in (
            "state_sync", "gop_position", "psnr", "lpips", "dists")} for r in rows if r["history"] == CLOSED])
    write_json(directory / "evaluation.json", {"summary": aggregate([r for r in rows if r["standard_cohort"]]),
                                               "extended_summary": aggregate(rows), "pixel_reads": reads,
                                               "seconds": duration,
                                               "scan_sha256": profile.wire.scan_hash.hex(),
                                               "model_sha256": profile.wire.model_hash.hex()})


def select_checkpoint(run, seed):
    store = CheckpointStore(CHECKPOINTS / f"seed{seed}", run.identity)
    directory = OUTPUT / f"seed{seed}"
    records = []
    for step, checkpoint in store.entries().items():
        step = int(step)
        if step <= run.config["phase_b_end"] or step % run.config["validate_every"]:
            continue
        rows = json_at(directory / "periodic" / f"step_{step:05d}" / "evaluation.json")["summary"]
        rd = mean(r["mse"] + 0.1 * r["lpips"] + 0.1 * r["dists"] + 0.02 * r["actual_bpp"] for r in rows)
        records.append({"step": step, "checkpoint_sha256": checkpoint["sha256"], "score": rd,
                        "scan": checkpoint["scan"]})
    if len(records) != (run.config["steps"] - run.config["phase_b_end"]) // run.config["validate_every"]:
        raise RuntimeError("missing scheduled Phase C native Validation")
    chosen = min(records, key=lambda row: (row["score"], row["step"]))
    selection = {"identity": run.identity, "steps": run.config["steps"], "chosen": chosen,
                 "criterion": "eight-budget mean of CLOSED MSE + 0.1 LPIPS + 0.1 DISTS + 0.02 actual bpp",
                 "candidates": records}
    write_json(directory / "selection.json", selection)


def load_selection(run, seed):
    status = json_at(OUTPUT / f"train_seed{seed}.json")
    selection = json_at(OUTPUT / f"seed{seed}" / "selection.json")
    if status["steps"] != run.config["steps"] or selection["identity"] != run.identity:
        raise RuntimeError("formal evaluation requires a completed 40,000-step trajectory and sealed selection")
    artifact, record = CheckpointStore(CHECKPOINTS / f"seed{seed}", run.identity).load(selection["chosen"]["step"])
    if record["sha256"] != selection["chosen"]["checkpoint_sha256"] or not artifact["scan"]["frozen"]:
        raise RuntimeError("selected checkpoint or frozen scan identity differs")
    load_model(run.codec, artifact)
    run.codec.eval().requires_grad_(False)
    return selection


def claim_test(run, selection):
    replica = json_at(OUTPUT / "train_seed2.json")
    validation = json_at(OUTPUT / "seed1" / "validation" / "evaluation.json")
    replica_selection = json_at(OUTPUT / "seed2" / "selection.json")
    hash_ = run.codec.profile().wire.model_hash.hex()
    if (run.identity["seed"] != 1 or replica["steps"] != run.config["steps"]
            or replica_selection["identity"] != (run.identity | {"seed": 2})
            or validation["model_sha256"] != hash_ or run.data.counts["test"] != 0):
        raise RuntimeError("held-out Test requires completed two-seed training and primary native Validation")
    path = OUTPUT / "test_release.json"
    seal = {"checkpoint": selection["chosen"], "config_sha256": run.identity["config_sha256"],
            "source_sha256": run.identity["source_sha256"], "model_sha256": hash_,
            "evaluation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "history": CLOSED, "budgets": list(BUDGETS), "gop": GOP, "test_pixel_reads_before_release": 0,
            "status": "CLAIMED"}
    with path.open("x") as handle:
        json.dump(seal, handle, indent=2)
    run.data.test_claim = seal
    return seal
