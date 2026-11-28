"""Paired LD / HT-S replay on the frozen native 1080p Validation cohort."""

import argparse
import json
import resource
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from scipy.interpolate import PchipInterpolator

from baselines.dcvc_uf.native import NativeNetworks
from baselines.dcvc_uf.official import NativeNAL
from baselines.dcvc_uf.units import NativeSchedule
from common.artifacts import write_csv, write_json
from common.inputs import inputs, source_clips
from common.integrity import tensor_digest
from common.performance import distribution, timed
from experiments.pht_tok.closed import official_decode, official_encode
from experiments.pht_tok.study import ROOT, SKIP, Quality, Source, aggregate_curves, ledger_row, uvg_source
from pht_tok.api import NATIVE_FRAMES
from pht_tok.bitstream import Mode
from pht_tok.codec import PHTDecoder, PHTEncoder, model_identifier
from third_party.DCVC.src.utils.common import set_torch_env
from third_party.DCVC.src.utils.transforms import rgb2ycbcr

SPLIT = {"ld": Mode.LD_MASK, "hts": Mode.MASK_FINE}
EXPECTED_READS = {"train": 0, "validation": 174, "test": 0}


@dataclass(frozen=True)
class Case:
    backend: str
    mode: Mode | None = None
    reference: int = 0

    @property
    def name(self):
        if self.mode is None:
            return f"native_{self.backend}"
        if self.mode == Mode.MASK:
            return "pht1_hts"
        return f"pht2_{self.backend}_r{self.reference}"

    def units(self, count):
        return list(NativeSchedule(NATIVE_FRAMES[self.backend], reset_interval=128).units(count))


class Endpoints:
    def __init__(self, weights, case):
        build = partial(NativeNetworks, weights, case.backend, SKIP)
        self.native_tx = build()
        self.native_rx = build()
        self.case = case
        if case.mode is not None:
            identity = model_identifier(weights, SKIP, case.backend)
            self.tx = PHTEncoder(build(), identity, backend=case.backend)
            self.rx = PHTDecoder(build(), identity, backend=case.backend)
        self.reset()

    def reset(self):
        self.writer, self.reader = NativeNAL(), NativeNAL()
        if self.case.mode is not None:
            self.tx.key = self.rx.key = None
            self.rx.session = None

    def encode_native(self, source, unit, qp):
        return official_encode(self.native_tx, source, unit, qp, self.writer)

    def decode_native(self, unit, wire):
        return official_decode(self.native_rx, unit, wire, self.reader)

    def encode_progressive(self, source, unit, qp):
        frames = source.yuv[unit.start:unit.start + unit.valid]
        if unit.intra:
            return self.tx.encode_intra(frames, qp, unit.start)
        return self.tx.encode_chunk_once(frames, qp, unit.start, self.case.reference,
                                         mode=self.case.mode, reset=unit.reset)

    def decode_progressive(self, payload):
        return self.rx.decode_prefix(payload).reconstruct()


def validation_sources():
    reader, plans = inputs()
    clips = source_clips(reader, plans["validation"])
    if reader.counts != EXPECTED_READS:
        raise RuntimeError("Validation pixel reads differ from the frozen 174-frame cohort")
    return [Source(x.clip.video, x.fps, x.rgb, rgb2ycbcr(x.rgb).half() - 0.5) for x in clips], reader.counts


def clone_frames(frames, unit, source):
    h, w = source.yuv.shape[-2:]
    return [frame[..., :h, :w].clone() for frame in frames[:unit.valid]]


def score_clip(source, units, physical, images, qp, method, stage, quality):
    per_frame = quality.score(source.rgb, images)
    return [ledger_row(source, unit, qp, method, stage, size,
                       per_frame[unit.start:unit.start + unit.valid])
            for unit, size in zip(units, physical, strict=True)]


def native_rd(source, endpoints, qp, quality):
    endpoints.reset()
    units = endpoints.case.units(len(source.rgb))
    packets = [endpoints.encode_native(source, u, qp) for u in units]
    frames = []
    for unit, packet in zip(units, packets, strict=True):
        frames.extend(clone_frames(endpoints.decode_native(unit, packet), unit, source))
    rows = score_clip(source, units, list(map(len, packets)), frames, qp, endpoints.case.name, 1, quality)
    return rows, [sha256(packet).hexdigest() for packet in packets]


def exact_tuple(left, right, label):
    if not all(torch.equal(a, b) for a, b in zip(left, right, strict=True)):
        raise ValueError(f"strict comparison failed for {label}")


def verify_full(ep, session, source, unit, expected, received):
    exact_tuple(received, clone_frames(expected, unit, source), "native reconstruction")
    native_state = ep.native_rx.video.proxy.audit_reference(unit.intra or unit.reset)
    exact_tuple(ep.rx.adapter.reference.tensors, native_state, "official reference tensors")
    if not unit.intra:
        z, y, indexes, keep = ep.native_rx.video.proxy.pht_symbols(False)
        pz, _, pindexes, pkeep = ep.rx.adapter.proxy.pht_symbols(False)
        exact_tuple((pz, session.residuals(), pindexes, pkeep), (z, y, indexes, keep),
                    "physically decoded native integers, indexes and skip mask")


def push_prefixes(ep, stream, unit, check_receiver):
    rx = ep.rx
    if check_receiver is not None and not unit.intra:
        check_receiver.key, check_receiver.adapter.reference = rx.key, rx.adapter.reference
    session = rx.begin_chunk()
    previous = 0
    images = []
    for stage, end in enumerate(stream.prefix_boundaries, 1):
        if stream.prefix(max_bytes=end) != stream.data[:end]:
            raise ValueError("legal budget changed physical prefix bytes")
        if check_receiver is None:
            session.append_bytes(stream.data[previous:end])
        else:
            cut = previous + (end - previous) * ((unit.start + stage) % 13) // 13
            session.append_bytes(stream.data[previous:cut], offset=previous)
            if session.decode_new_layers() != stage - 1:
                raise ValueError("incomplete entropy record advanced the prefix")
            session.append_bytes(stream.data[previous:cut], offset=previous)
            session.append_bytes(stream.data[cut:end], offset=cut)
        if session.decode_new_layers() != stage:
            raise ValueError("a complete prefix failed to advance exactly one stage")
        output = session.reconstruct()
        if check_receiver is not None:
            independent = check_receiver.decode_prefix(stream.data[:end], commit=False)
            exact_tuple(output, independent.reconstruct(), "independent versus incremental display")
        images.append(output)
        previous = end
    return session, images


def progressive_rd(source, ep, qp, quality, check_receiver=None, native_hashes=None):
    ep.reset()
    if check_receiver is not None:
        check_receiver.key = None
    case = ep.case
    units = case.units(len(source.rgb))
    layers = case.mode.layers
    sizes, images = [[] for _ in range(layers)], [[] for _ in range(layers)]
    checks = []
    for number, unit in enumerate(units):
        packet = ep.encode_progressive(source, unit, qp)
        session, display = push_prefixes(ep, packet, unit, check_receiver)
        exact_tuple(ep.tx.adapter.reference.tensors, ep.rx.adapter.reference.tensors, "independent closed reference")
        native_bytes = native_sha = None
        if native_hashes is not None:
            wire = ep.encode_native(source, unit, qp)
            native_sha = sha256(wire).hexdigest()
            if native_sha != native_hashes[number]:
                raise ValueError("paired native wire changed between independent replays")
            verify_full(ep, session, source, unit, ep.decode_native(unit, wire), display[-1])
            native_bytes = len(wire)
        checks.append({"source": source.name, "qp": qp, "method": case.name, "start": unit.start,
                       "frames": unit.valid, "intra": unit.intra, "refresh": unit.reset,
                       "stages": list(packet.prefix_boundaries), "native_bytes": native_bytes,
                       "native_sha256": native_sha,
                       "ref_bytes": packet.prefix_boundaries[0 if unit.intra else case.reference - 1],
                       "wire_sha256": sha256(packet.data).hexdigest(), "header": packet.ledger["header_bytes"],
                       "record_metadata": packet.ledger["layer_header_bytes"], "payload_sizes": packet.sizes,
                       "sender_receiver_exact": True, "native_exact": native_bytes is not None,
                       "independent_prefix_exact": check_receiver is not None})
        for stage in range(layers):
            index = 0 if unit.intra else stage
            sizes[stage].append(packet.prefix_boundaries[index])
            images[stage].extend(display[index])
    stages = range(1, layers + 1) if case.reference < layers else (layers,)
    rows = [row for stage in stages for row in score_clip(
        source, units, sizes[stage - 1], images[stage - 1], qp, case.name, stage, quality)]
    return rows, checks


def validation(weights, output, qps, backends=tuple(SPLIT)):
    sources, reads = validation_sources()
    quality = Quality()
    rd_rows, checks = [], []
    for backend in backends:
        mode = SPLIT[backend]
        native = Endpoints(weights, Case(backend))
        equivalent = Endpoints(weights, Case(backend, mode, mode.layers))
        closed = Endpoints(weights, Case(backend, mode, 1))
        fresh = PHTDecoder(NativeNetworks(weights, backend, SKIP), closed.tx.model_id, backend=backend)
        for qp in qps:
            for source in sources:
                part = output / "parts" / f"{backend}-{qp}-{source.name.split(':')[-1]}.json"
                if part.is_file():
                    measured = json.loads(part.read_text())
                else:
                    rows, hashes = native_rd(source, native, qp, quality)
                    full_rows, full_checks = progressive_rd(source, equivalent, qp, quality, None, hashes)
                    preview_rows, preview_checks = progressive_rd(source, closed, qp, quality,
                                                                  fresh if qp == 32 else None)
                    measured = {"rd": rows + full_rows + preview_rows, "prefix": full_checks + preview_checks}
                    write_json(part, measured)
                rd_rows.extend(measured["rd"])
                checks.extend(measured["prefix"])
                print("Validation", backend, qp, source.name, flush=True)
    validation_reports(output, qps, reads, rd_rows, checks)


def validation_reports(output, qps, reads, rd_rows, checks):
    curves = aggregate_curves(rd_rows)
    for row in curves:
        units = sum(item["method"] == row["method"] and item["qp"] == row["qp"] and item["stage"] == row["stage"]
                    for item in rd_rows)
        row["bytes_per_frame"] = row["bytes"] / row["frames"]
        row["bytes_per_coding_unit"] = row["bytes"] / units
    write_csv(output / "rate_distortion.csv", curves)
    write_json(output / "validation_units.json", {"reads": reads, "units": rd_rows})
    write_json(output / "prefix_validation.json", {"reads": reads, "units": checks,
        "strict_prefix": True, "fragmented_and_duplicate_receive_qp": 32})
    write_json(output / "native_equivalence.json", {"reads": reads, "fp16_atol": 0, "fp16_rtol": 0,
        "coding_units": {name: [r for r in checks if r["method"] == f"pht2_{name}_r8"] for name in SPLIT}})
    references = [{"backend": name, "qp": qp, "mode": method, "coding_units": len(group),
                   "reference_bytes": sum(r["ref_bytes"] for r in group),
                   "full_bytes": sum(r["stages"][-1] for r in group),
                   "exact": all(r["sender_receiver_exact"] for r in group)}
                  for name in SPLIT for qp in qps for method in (1, 8)
                  if (group := [r for r in checks if r["method"] == f"pht2_{name}_r{method}" and r["qp"] == qp])]
    write_json(output / "reference_validation.json", {"aggregates": references, "test_pixel_reads": reads["test"]})
    write_rd_analysis(output, curves, rd_rows)


def bd(left, right, quality, mode):
    # Each curve is cut to its shared measured domain before integrating.
    def knots(rows):
        rates = np.log([r["mbps"] for r in rows])
        values = np.array([r[quality] for r in rows])
        return (rates, values) if mode == "quality" else (values, rates)

    x, y = knots(left)
    u, v = knots(right)
    if min(len(x), len(u)) < 4:
        return None
    a, b = np.argsort(x), np.argsort(u)
    x, y, u, v = x[a], y[a], u[b], v[b]
    lower, upper = max(x[0], u[0]), min(x[-1], u[-1])
    if lower >= upper or np.any(np.diff(x) == 0) or np.any(np.diff(u) == 0):
        return None
    diff = (PchipInterpolator(u, v).integrate(lower, upper) -
            PchipInterpolator(x, y).integrate(lower, upper)) / (upper - lower)
    return {"shared_interval": [float(lower), float(upper)],
            "delta": float(diff if mode == "quality" else 100 * np.expm1(diff))}


def write_rd_analysis(output, curves, rows):
    ablation = []
    analysis = {}
    for backend, mode in SPLIT.items():
        native = [r for r in curves if r["method"] == f"native_{backend}"]
        closed = [r for r in curves if r["method"] == f"pht2_{backend}_r1"]
        full = [r for r in curves if r["method"] == f"pht2_{backend}_r{mode.layers}"]
        if not native:
            continue
        for n, c in zip(native, full, strict=True):
            ablation.append({"backend": backend, "qp": n["qp"], "native_bytes": n["bytes"], "full_bytes": c["bytes"],
                             "full_extra_percent": 100 * (c["bytes"] / n["bytes"] - 1),
                             "quality_delta_db": c["global_psnr"] - n["global_psnr"]})
        analysis[backend] = {
            str(stage): {metric: {"equal_quality_bd_rate_percent": bd(native, points, metric, "rate"),
                                 "equal_rate_bd_quality": bd(native, points, metric, "quality")}
                         for metric in ("global_psnr", "lpips", "dists", "ms-ssim")}
            for stage in range(1, mode.layers + 1)
            if (points := [r for r in closed if r["stage"] == stage])
        }
        analysis[backend]["nonmonotonic_adjacent_frames"] = reversals(rows, f"pht2_{backend}_r1")
        analysis[backend]["paired_qp32_full_minus_native_db"] = paired_interval(rows, backend)
    write_csv(output / "ablation.csv", ablation)
    write_json(output / "rate_distortion_analysis.json", analysis)


def reversals(rows, method):
    metrics = {"psnr": 1, "lpips": -1, "dists": -1, "ms-ssim": 1}
    reversals = dict.fromkeys(metrics, 0)
    indexed = {(r["source"], r["qp"], r["start"], r["stage"]): r for r in rows
               if r["method"] == method and not r["intra"]}
    pairs = 0
    for (name, qp, start, stage), row in indexed.items():
        previous = indexed.get((name, qp, start, stage - 1))
        if previous is not None:
            for a, b in zip(previous["frame_metrics"], row["frame_metrics"], strict=True):
                pairs += 1
                for metric, direction in metrics.items():
                    reversals[metric] += int((b[metric] - a[metric]) * direction < 0)
    return {"pairs": pairs, "decreases": reversals}


def paired_interval(rows, backend):
    clips = {}
    for row in rows:
        if row["qp"] == 32 and (row["method"], row["stage"]) in (
            (f"native_{backend}", 1), (f"pht2_{backend}_r1", 8)
        ):
            clips.setdefault((row["source"], row["method"]), []).extend(x["psnr"] for x in row["frame_metrics"])
    sources = sorted({name for name, _ in clips})
    if not sources:
        return None
    values = np.array([np.mean(clips[name, f"pht2_{backend}_r1"]) - np.mean(clips[name, f"native_{backend}"])
                       for name in sources])
    rng = np.random.default_rng(272026)
    draws = rng.choice(values, (10000, len(values)), replace=True).mean(axis=1)
    return {"unit": "video", "videos": len(sources), "per_video_db": dict(zip(sources, values.tolist(), strict=True)),
            "bootstrap_95_db": np.percentile(draws, [2.5, 97.5]).tolist()}


def apply_fragment(session, data):
    session.append_bytes(data)
    session.decode_new_layers()
    session.reconstruct()


def incremental_timing(ep, packet):
    session = ep.rx.begin_chunk()
    previous = 0
    times = []
    for end in packet.prefix_boundaries:
        _, elapsed = timed(apply_fragment, session, packet.data[previous:end])
        times.append(elapsed)
        previous = end
    return sum(times), times[0]


def benchmark_unit(ep, source, unit, qp, incremental):
    mode = ep.case.mode
    if mode is None:
        raw, enc = timed(ep.encode_native, source, unit, qp)
        _, dec = timed(ep.decode_native, unit, raw)
        return len(raw), enc, dec, dec, 0
    packet, enc = timed(ep.encode_progressive, source, unit, qp)
    if incremental:
        dec, first = incremental_timing(ep, packet)
    else:
        _, dec = timed(ep.decode_progressive, packet.data)
        first = dec
    overhead = packet.ledger["header_bytes"] + packet.ledger["layer_header_bytes"]
    return len(packet.data), enc, dec, first, overhead


def kernel_time(ep, source, units, qp):
    from torch.profiler import ProfilerActivity, profile

    ep.reset()
    for unit in units[:5]:
        benchmark_unit(ep, source, unit, qp, False)

    def totals(prof):
        events = [event for event in prof.events() if str(event.device_type).endswith("CUDA")]
        kernels = [event for event in events if not event.name.startswith(("Memcpy", "Memset"))]
        return {"kernels": len(kernels), "kernel_sum_ms": sum(e.device_time_total for e in kernels) / 1000}

    unit = units[5]
    native = ep.case.mode is None
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as encoding:
        payload = (ep.encode_native if native else ep.encode_progressive)(source, unit, qp)
        torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as decoding:
        if native:
            ep.decode_native(unit, payload)
        else:
            ep.decode_progressive(payload.data)
        torch.cuda.synchronize()
    return {"encode": totals(encoding), "decode": totals(decoding),
            "definition": "CUPTI CUDA kernel-duration sum; overlapping kernels counted individually"}


def runtime_rows(ep, source, units, qp, incremental, repetitions):
    rows = []
    torch.cuda.reset_peak_memory_stats()
    for repeat in range(repetitions):
        ep.reset()
        for i, unit in enumerate(units):
            raw, enc, dec, first, meta = benchmark_unit(ep, source, unit, qp, incremental)
            wait = (unit.valid - 1) * 1000 / source.fps
            rows.append({"method": ep.case.name + ("_incremental" if incremental else ""),
                         "repeat": repeat, "start": unit.start, "frames": unit.valid,
                         "intra": unit.intra, "warmup": i <= 4, "wire_bytes": raw,
                         "envelope_bytes": meta, "encode_ms": enc, "decode_ms": dec,
                         "serial_ms": enc + dec, "capture_wait_ms": wait, "first_display_ms": wait + enc + first,
                         "sampled_process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                         "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                         "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved()})
    return rows


def summarize_runtime(rows):
    label = rows[0]["method"]
    steady = [r for r in rows if not r["warmup"]]
    frames = sum(r["frames"] for r in steady)
    summary = {"method": label, "frames_after_warmup": frames, "bytes": sum(r["wire_bytes"] for r in steady),
               "envelope_bytes": sum(r["envelope_bytes"] for r in steady),
               "peak_gpu_bytes": max(r["peak_cuda_allocated_bytes"] for r in rows),
               "process_high_water_rss_kib": max(r["sampled_process_peak_rss_kib"] for r in rows)}
    for name in ("encode", "decode", "serial", "capture_wait", "first_display"):
        samples = [r[f"{name}_ms"] for r in steady]
        summary.update(distribution(samples, f"{name}_ms"))
        if name in ("encode", "decode", "serial"):
            summary[f"{name}_fps"] = 1000 * frames / sum(samples)
    return summary


def runtime(weights, output, uvg, repetitions):
    source = uvg_source(uvg, 129)
    methods = [Case("ld"), Case("ld", Mode.LD_MASK, 1), Case("hts"),
               Case("hts", Mode.MASK, 2), Case("hts", Mode.MASK_FINE, 1)]
    details, metrics, gpu = [], [], {}
    for case in methods:
        ep = Endpoints(weights, case)
        units = case.units(len(source.rgb))
        for incremental in (False, True) if case.mode else (False,):
            rows = runtime_rows(ep, source, units, 32, incremental, repetitions)
            details.extend(rows)
            metrics.append(summarize_runtime(rows))
        gpu[case.name] = kernel_time(ep, source, units, 32)
        print("Runtime", case.name, flush=True)
    original = json.loads((ROOT / "results/pht_tok/baseline_replay/profiling.json").read_text())
    replay = original["mask_kref2"]["full"]
    metrics.append({"method": "original_pht1_hts_at_54674cc",
                    **{name: float(np.mean([r[name] for r in replay]))
                       for name in ("encode_fps", "decode_fps", "serial_fps")}})
    write_csv(output / "throughput.csv", metrics)
    write_csv(output / "latency.csv", details)
    write_json(output / "runtime_diagnostics.json", {"capture_fps": source.fps, "resolution": [1080, 1920],
        "repetitions": repetitions, "warmup_p_units": 4, "gpu_kernel_profile": gpu,
        "wall_scope": "CUDA-synchronized encoder/decoder wall time, entropy and serialization, preloaded input, batch 1",
        "cpu_memory_scope": "Linux process high-water RSS sampled at completed units, cumulative within the runtime process"})


def official_control(weights, output, qps):
    sources, reads = validation_sources()
    ledger = {}
    for backend in SPLIT:
        ep = Endpoints(weights, Case(backend))
        for source in sources:
            for qp in qps:
                ep.reset()
                units = ep.case.units(len(source.rgb))
                packets = [ep.encode_native(source, unit, qp) for unit in units]
                ledger[f"{backend}/{source.name}/{qp}"] = [{
                    "wire_sha256": sha256(packet).hexdigest(),
                    "reconstruction_sha256": tensor_digest(clone_frames(ep.decode_native(unit, packet), unit, source)),
                } for unit, packet in zip(units, packets, strict=True)]
    write_json(output / "native_stock_wire.json", {"reads": reads, "sha256": ledger})


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=("validation", "runtime", "network", "official-control", "adaptok"))
    parser.add_argument("--weights", "--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qps", nargs="+", type=int, choices=range(64), default=[16, 32, 48, 63])
    parser.add_argument("--backend", choices=SPLIT, nargs="+", default=tuple(SPLIT))
    parser.add_argument("--uvg", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.study == "runtime" and (args.uvg is None or args.repetitions < 1):
        parser.error("runtime needs --uvg and at least one repetition")
    if args.study == "adaptok":
        from experiments.pht_tok.adaptok_control import adaptok
        adaptok(args.weights, args.output)
        return
    set_torch_env()
    with torch.cuda.stream(torch.cuda.Stream()):
        if args.study == "validation":
            validation(args.weights, args.output, sorted(set(args.qps)), tuple(dict.fromkeys(args.backend)))
        elif args.study == "runtime":
            runtime(args.weights, args.output, args.uvg, args.repetitions)
        elif args.study == "official-control":
            official_control(args.weights, args.output, sorted(set(args.qps)))
        else:
            from experiments.pht_tok.network_dual import network
            network(args.weights, args.output)


if __name__ == "__main__":
    main()
