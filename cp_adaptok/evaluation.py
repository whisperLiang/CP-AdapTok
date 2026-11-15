import io
import struct
import time

import numpy as np
import torch

from cp_adaptok.data import check_splits, read_video, save_rgb
from cp_adaptok.metrics import MetricSuite, summarize_latency
from cp_adaptok.runtime import metadata, resolve, write_json
from cp_adaptok.streaming.online_allocator import OnlineTokenAllocator, RDPoint, offline_oracle
from cp_adaptok.streaming.packetizer import validate_budgets
from cp_adaptok.streaming.receiver import CodecDecoder
from cp_adaptok.streaming.sender import CodecEncoder, synchronize


def serialize_prefix(encoded, k):
    out = io.BytesIO()
    for record in (encoded.header_bytes, *encoded.prefix_packets(k)):
        out.write(struct.pack(">I", len(record)))
        out.write(record)
    return out.getvalue()


def _online_choices(allocator, measured, predicted, frame, config):
    fixed = min(measured, key=lambda p: abs(p.tokens - config.get("fixed_k", 512)))
    playback_delay = config.get("playback_delay_s", 1 / frame["fps"])
    remaining = max(0.0, playback_delay - frame["encode_timing_ms"]["encode_total_ms"] / 1000)
    options = {"bandwidth_bps": config.get("bandwidth_bps", 500_000), "deadline_s": remaining}
    choices = {}
    actual = {p.tokens: p for p in measured}
    for mode in ("fixed", "rate-only", "scorer"):
        selected = allocator.allocate(predicted, mode=mode, fixed_k=fixed.tokens, **options)
        choices[mode] = (
            {
                "tokens": selected.tokens,
                "bytes": selected.bytes,
                "actual_mse": actual[selected.tokens].distortion,
                "predicted_mse": selected.distortion,
            }
            if selected
            else {"status": "base_wait_required"}
        )
    return choices, fixed.bytes


def _sequence_allocations(name, frames, config, allocator):
    measured, output, fixed_total = [], [], 0
    for frame in frames:
        points = [RDPoint(r["tokens"], r["total_bytes"], r["mse"]) for r in frame["prefixes"]]
        predicted = [RDPoint(r["tokens"], r["total_bytes"], r["predicted_mse"]) for r in frame["prefixes"]]
        choices, fixed_bytes = _online_choices(allocator, points, predicted, frame, config)
        fixed_total += fixed_bytes
        measured.append(points)
        output.append(
            {
                "sequence": name,
                "frame_id": frame["frame_id"],
                "online": choices,
                "priority": allocator.enhancement_priority(predicted),
            }
        )
    for row, point in zip(output, offline_oracle(measured, fixed_total)):
        row["offline_oracle"] = {
            "tokens": point.tokens,
            "bytes": point.bytes,
            "actual_mse": point.distortion,
            "noncausal": True,
        }
    return output


def allocation_comparison(frame_records, config, k_ref):
    grouped = {}
    for record in frame_records:
        grouped.setdefault(record["sequence"], []).append(record)
    allocator = OnlineTokenAllocator(k_ref)
    output = []
    for name, frames in grouped.items():
        output.extend(_sequence_allocations(name, frames, config, allocator))
    return output


class FrameEvaluator:
    def __init__(self, model, config, metrics, budgets, output, compare_cache):
        self.model, self.metrics, self.budgets = model, metrics, budgets
        self.output, self.compare_cache = output, compare_cache
        self.save_frames = config.get("evaluation", {}).get("save_frames", True)
        self.sender = CodecEncoder(model, config["evaluation"]["intra_period"])
        self.receiver = CodecDecoder(model)
        self.previous = {}

    def _cached_metrics(self, k, dense, original):
        synchronize(self.model.device)
        start = time.perf_counter()
        cached = self.receiver.decode_prefix(k, use_cache=True)
        synchronize(self.model.device)
        return {
            "cache_decode_ms": (time.perf_counter() - start) * 1000,
            "cache_max_error": (cached - dense).abs().max().item(),
            "cache_metrics": self.metrics(original, cached),
            **self.receiver.last_timing_ms,
        }

    def _prefix_row(self, original, encoded, k, fps, t, directory):
        synchronize(self.model.device)
        start = time.perf_counter()
        dense = self.receiver.decode_prefix(k, use_cache=False)
        synchronize(self.model.device)
        elapsed_ms = (time.perf_counter() - start) * 1000
        values = self.metrics(original, dense)
        row = {
            "tokens": k,
            **values,
            "predicted_mse": encoded.predicted_mse[k],
            "prediction_error": encoded.predicted_mse[k] - values["mse"],
            "dense_decode_ms": elapsed_ms,
            "total_bytes": encoded.bytes_at(k),
            "codec_bytes": encoded.bytes_at(k, transport=False),
            "bpp": encoded.bytes_at(k) * 8 / (original.shape[-1] * original.shape[-2]),
            "kbps": encoded.bytes_at(k) * 8 * fps / 1000,
        }
        if self.compare_cache:
            row.update(self._cached_metrics(k, dense, original))
        if k in self.previous:
            prev_original, prev_dense = self.previous[k]
            row["temporal_error"] = ((original - prev_original) - (dense - prev_dense)).square().mean().item()
        self.previous[k] = original, dense
        if self.save_frames:
            save_rgb(directory / f"frame{t:03d}_k{k}.png", dense)
        return row

    def _base(self, encoded):
        synchronize(self.model.device)
        start = time.perf_counter()
        count = self.model.config.k_ref // self.model.config.packet_tokens
        self.receiver.append_packets(encoded.packets[:count])
        self.receiver.decode_base()
        synchronize(self.model.device)
        return count, (time.perf_counter() - start) * 1000

    @torch.no_grad()
    def frame(self, rgb, t, spec):
        name = resolve(spec["path"]).stem
        original = rgb[None].to(self.model.device)
        encoded = self.sender.encode_frame(original, budgets=self.budgets)
        self.receiver.start_frame(encoded.header_bytes)
        count, base_ms = self._base(encoded)
        fingerprint = self.receiver.get_state().fingerprint()
        directory = self.output / name
        directory.mkdir(parents=True, exist_ok=True)
        rows = []
        for k in self.budgets:
            end = k // self.model.config.packet_tokens
            self.receiver.append_packets(encoded.packets[count:end])
            count = end
            rows.append(self._prefix_row(original, encoded, k, spec["fps"], t, directory))
        if self.save_frames:
            save_rgb(directory / f"frame{t:03d}_original.png", original)
        (directory / f"frame{t:03d}.cpat").write_bytes(serialize_prefix(encoded, self.budgets[-1]))
        fingerprints = {
            fingerprint,
            self.receiver.get_state().fingerprint(),
            self.sender.get_state().fingerprint(),
        }
        if len(fingerprints) != 1:
            raise RuntimeError("reference drift detected")
        return {
            "sequence": name,
            "source": spec,
            "frame_id": t,
            "frame_type": encoded.header.frame_type,
            "shape": list(original.shape),
            "fps": spec["fps"],
            "prefixes": rows,
            "base_decode_ms": base_ms,
            "encode_timing_ms": encoded.timing_ms,
            "byte_counts": self.receiver.stream.byte_counts(),
            "state_fingerprint": fingerprint,
            "reconstructions": self.receiver.reconstructions,
            "context_builds": self.receiver.context_builds,
        }


def budget_summary(records, budgets):
    summaries = []
    for i, k in enumerate(budgets):
        row = {"tokens": k}
        for name in ("mse", "psnr", "ms-ssim", "lpips", "dists", "bpp", "kbps"):
            values = [r["prefixes"][i][name] for r in records if r["prefixes"][i].get(name) is not None]
            row[name] = float(np.mean(values)) if values else None
        summaries.append(row)
    return summaries


def _quality_summary(records, compare_cache):
    pairs = [
        (a["mse"], b["mse"]) for frame in records for a, b in zip(frame["prefixes"], frame["prefixes"][1:])
    ]
    drops = sum(b > a for a, b in pairs)
    cache_passed = compare_cache and all(
        row["cache_max_error"] <= 1e-5 for frame in records for row in frame["prefixes"]
    )
    return {
        "quality_drop_pairs": drops,
        "quality_compared_pairs": len(pairs),
        "quality_drop_ratio": drops / len(pairs) if pairs else 0,
        "cache_equivalence_verified": cache_passed,
    }


def evaluate_sequences(
    model,
    config,
    split="test",
    budgets=None,
    metric_names=None,
    status="unknown",
    output=None,
    compare_cache=True,
    max_frames=None,
):
    check_splits(config["data"])
    output = resolve(output or config["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    budgets = config["evaluation"]["budgets"] if budgets is None else budgets
    validate_budgets(budgets, model.config)
    metrics = MetricSuite(model.device, metric_names or config["evaluation"]["metrics"])
    records = []
    for configured_spec in config["data"][split]:
        spec = dict(configured_spec)
        if max_frames:
            spec["frames"] = min(spec["frames"], max_frames)
        evaluator = FrameEvaluator(model, config, metrics, budgets, output, compare_cache)
        records.extend(evaluator.frame(rgb, t, spec) for t, rgb in enumerate(read_video(spec)))
    if not records:
        raise ValueError("evaluation split is empty")
    report = metadata(config, status)
    report.update(
        artifact=getattr(model, "artifact_provenance", None),
        model_id=model.model_id,
        codebook_id=model.vq.codebook_id(),
        real_entropy_bitstream=True,
        split=split,
        sample_count=len(records),
        records=records,
        metric_errors=metrics.errors,
        trained_rd_validated=False,
        interpretation="engineering measurements; no generalization claim",
        link_layer_overhead="not captured; codec records and actual TCP application framing counted",
        allocation=allocation_comparison(records, config.get("allocation", {}), model.config.k_ref),
        budget_summary=budget_summary(records, budgets),
        **_quality_summary(records, compare_cache),
    )
    write_json(output / "evaluation.json", report)
    return report


class PipelineBenchmark:
    def __init__(self, model, budgets, use_cache):
        validate_budgets(budgets, model.config)
        self.model, self.budgets, self.use_cache = model, budgets, use_cache
        self.sender, self.receiver = CodecEncoder(model), CodecDecoder(model)

    def reset(self):
        self.sender.reset()
        self.receiver.reset()
        if self.model.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.model.device)

    @torch.no_grad()
    def frame(self, frame):
        synchronize(self.model.device)
        start = time.perf_counter()
        encoded = self.sender.encode_frame(frame[None].to(self.model.device), budgets=self.budgets)
        self.receiver.start_frame(encoded.header_bytes)
        count = 0
        before = self.receiver.reconstructions
        stages = []
        for k in self.budgets:
            end = k // self.model.config.packet_tokens
            self.receiver.append_packets(encoded.packets[count:end])
            count = end
            self.receiver.decode_prefix(k, use_cache=self.use_cache)
            stages.append({"tokens": k, **self.receiver.last_timing_ms})
        synchronize(self.model.device)
        total_ms = (time.perf_counter() - start) * 1000
        if self.sender.get_state().fingerprint() != self.receiver.get_state().fingerprint():
            raise RuntimeError("benchmark reference drift")
        return {
            "frame_type": encoded.header.frame_type,
            "pipeline_ms": total_ms,
            "encoder": encoded.timing_ms,
            "prefix_stages": stages,
            "bytes": encoded.bytes_at(self.budgets[-1]),
            "reconstructions": self.receiver.reconstructions - before,
        }


def _benchmark_report(rows, model, resolution, budgets):
    overall = summarize_latency([r["pipeline_ms"] for r in rows])
    inter = [r["pipeline_ms"] for r in rows if r["frame_type"] == "P"]
    gpu = model.device.type == "cuda"
    return {
        "overall": overall,
        "steady_inter": summarize_latency(inter) if inter else None,
        "frames": rows,
        "resolution": list(resolution),
        "display_points": budgets,
        "real_entropy_bitstream": True,
        "includes_sender_local_base_decode": True,
        "transport": "in-process serialized records; socket demo measured separately",
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(model.device) if gpu else None,
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(model.device) if gpu else None,
        "target_30fps_met": overall["fps"] >= 30,
        "target_60fps_met": overall["fps"] >= 60,
    }


def benchmark_pipeline(model, config, resolution, frames, warmup, budgets, use_cache=True):
    if frames < 1 or warmup < 0:
        raise ValueError("invalid benchmark sample counts")
    spec = dict(config["data"]["test"][0], frames=max(frames, warmup, 1))
    clip = read_video(spec)
    width, height = resolution
    if clip.shape[-2:] != (height, width):
        raise ValueError(
            "benchmark source must have the requested native resolution; provide a matching source"
        )
    runner = PipelineBenchmark(model, budgets, use_cache)
    for frame in clip[:warmup]:
        runner.frame(frame)
    runner.reset()
    rows = [runner.frame(frame) for frame in clip[:frames]]
    return _benchmark_report(rows, model, resolution, budgets)
