"""Exercise real sender/receiver R1 state and native wall-clock throughput."""

from functools import partial
from statistics import mean
from time import process_time

import numpy as np
import torch

from baselines.dcvc_uf.official import NativeNAL
from baselines.dcvc_uf.units import NativeSchedule, Unit
from common.artifacts import write_json
from common.performance import timed
from pht_tok.bitstream import Chunk, ChunkHeader, Mode, StreamReader
from pht_tok.codec import MissingReference

PATTERNS = ("base", "mid", "full", "alternating", "random", "burst", "late_enhancement",
            "lost_enhancement", "lost_reference", "bad_reference_version")


def budget_pattern(pattern: str, k_ref: int, chunks: int) -> list[int]:
    rng = np.random.default_rng(272026)
    choices = {
        "base": [k_ref] * chunks,
        "mid": [max(k_ref, 3)] * chunks,
        "full": [4] * chunks,
        "alternating": [k_ref if i % 2 == 0 else 4 for i in range(chunks)],
        "random": rng.integers(k_ref, 5, chunks).tolist(),
        "burst": [k_ref if 3 <= i <= 7 else 4 for i in range(chunks)],
        "late_enhancement": [k_ref if i == 4 else 4 for i in range(chunks)],
        "lost_enhancement": [k_ref if i == 4 else 4 for i in range(chunks)],
        "lost_reference": [k_ref - 1 if i == 4 else 4 for i in range(chunks)],
        "bad_reference_version": [4] * chunks,
    }
    return choices[pattern]


def summarized_speed(rows: list[dict], fps: float) -> dict:
    steady = [row for row in rows if not row["intra"]][4:]
    enc = [row["encode_ms"] for row in steady]
    dec = [row["decode_ms"] for row in steady]
    serial = [a + b for a, b in zip(enc, dec, strict=True)]
    frames = sum(row["valid_frames"] for row in steady)
    return {"valid_frames": frames, "encode_fps": frames * 1000 / sum(enc),
            "decode_fps": frames * 1000 / sum(dec), "serial_fps": frames * 1000 / sum(serial),
            "assembly_wait_ms": 7000 / fps,
            "amortized_encode_ms_per_frame": sum(enc) / frames,
            "amortized_decode_ms_per_frame": sum(dec) / frames,
            "encode_ms_per_chunk_p50_p95_p99": np.percentile(enc, [50, 95, 99]).tolist(),
            "decode_ms_per_chunk_p50_p95_p99": np.percentile(dec, [50, 95, 99]).tolist(),
            "serial_ms_per_chunk_p50_p95_p99": np.percentile(serial, [50, 95, 99]).tolist()}


def official_encode(stock, source, unit, qp, writer):
    frames = source.yuv[unit.start:unit.start + unit.valid]
    return writer.encode(stock.encode_centered(frames, unit, qp))


def official_decode(stock, unit, wire, reader):
    base = reader.decode(wire, unit.start)
    return stock.decode(base)


def official_profile(source, stock, qp):
    units = list(NativeSchedule(8, reset_interval=128).units(len(source.yuv)))
    writer, reader = NativeNAL(), NativeNAL()
    rows = []
    start_cpu = process_time()
    torch.cuda.reset_peak_memory_stats()
    streams = [timed(official_encode, stock, source, unit, qp, writer) for unit in units]
    for unit, (raw, encode_ms) in zip(units, streams, strict=True):
        _, decode_ms = timed(official_decode, stock, unit, raw, reader)
        rows.append(dict(start=unit.start, intra=unit.intra, valid_frames=unit.valid, bytes=len(raw),
                         encode_ms=encode_ms, decode_ms=decode_ms))
    return profile_report(rows, source, process_time() - start_cpu)


def profile_report(rows, source, cpu_seconds):
    speed = summarized_speed(rows, source.fps)
    return speed | {"source": source.name, "fps": source.fps, "per_unit": rows,
                    "process_cpu_seconds": cpu_seconds, "peak_gpu_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "peak_gpu_reserved_bytes": torch.cuda.max_memory_reserved(),
                    "warmup_p_chunks": 4, "wall_timing": "GPU-synchronized, batch 1, source preloaded",
                    "explicit_timing_cuda_sync_calls_per_unit": 4}


def result_quality(rows):
    per_frame = [item for unit in rows for item in unit.get("quality", [])]
    if not per_frame:
        return {}
    stats = {metric: mean(item[metric] for item in per_frame) for metric in per_frame[0]}
    return stats | {"global_psnr": float(-10 * np.log10(stats["mse"])), "frames": len(per_frame)}


def corrupt_reference_version(raw: bytes) -> bytes:
    reader = StreamReader()
    records = reader.append_bytes(raw)
    h = reader.header
    assert h is not None
    bad = ChunkHeader(h.model, h.chunk_id, h.parent_id, h.reference_version + 1, h.height,
                      h.width, h.qp, h.valid_frames, h.mode, h.k_ref, h.reset_memory)
    return Chunk.pack(bad, records).data


def late_probe(rx, session, remainder):
    if remainder:
        # A new read session owns the scratch only after the prior display deadline.
        next_read = rx.begin_chunk()
        try:
            session.append_bytes(remainder)
        except ValueError:
            return len(remainder)
        if next_read is not session:
            raise ValueError("expired enhancement changed receiver state")
    return 0


def run_closed(source, qp, tx, rx, mode, k_ref, pattern, quality=None, render_incrementally=False):
    tx.key = rx.key = None
    units = list(NativeSchedule(8, reset_interval=128).units(len(source.yuv)))
    budgets = budget_pattern(pattern, k_ref, len(units) - 1)
    rows = []
    start_cpu = process_time()
    torch.cuda.reset_peak_memory_stats()
    first, enc_ms = timed(tx.encode_intra, source.yuv[:1], qp, 0)
    first_session, dec_ms = timed(rx.decode_prefix, first.data)
    assert tx.adapter.reference is not None and rx.adapter.reference is not None
    if tx.adapter.reference.digest() != rx.adapter.reference.digest():
        raise ValueError("independent I-frame reference drift")
    rows.append(dict(start=0, intra=True, valid_frames=1, bytes=len(first.data),
                     encode_ms=enc_ms, decode_ms=dec_ms, reference_hash=rx.adapter.reference.digest()))
    if quality:
        rows[0]["quality"] = quality.score(source.rgb[:1], first_session.reconstruct())
    for p_index, (unit, wanted) in enumerate(zip(units[1:], budgets, strict=True)):
        chunk_id = unit.start
        if pattern in ("lost_reference", "bad_reference_version") and p_index == 6:
            refresh, enc_ms = timed(tx.encode_intra, source.yuv[chunk_id:chunk_id + 1], qp, chunk_id)
            _, dec_ms = timed(rx.decode_prefix, refresh.data)
            rows.append(dict(start=chunk_id, intra=True, refresh=True, valid_frames=1, bytes=len(refresh.data),
                             encode_ms=enc_ms, decode_ms=dec_ms, reference_hash=rx.adapter.reference.digest()))
            unit = Unit(unit.start + 1, unit.valid - 1, unit.coded, False, unit.reset)
            chunk_id += 1
        sent, encoder_ms = timed(partial(tx.encode_chunk_once, mode=mode, reset=unit.reset),
                                 source.yuv[unit.start:unit.start + unit.valid], qp, chunk_id, k_ref)
        prefix = sent.data[:sent.prefix_boundaries[wanted - 1]] if wanted else b""
        if pattern == "bad_reference_version" and p_index == 4:
            prefix = corrupt_reference_version(sent.data)
        row = dict(start=unit.start, intra=False, valid_frames=unit.valid, bytes=len(prefix), stage=wanted,
                   encode_ms=encoder_ms, decode_ms=0.0, full_bytes=len(sent.data), reset=unit.reset,
                   ref_bytes=sent.prefix_boundaries[k_ref - 1], prefix_boundaries=list(sent.prefix_boundaries))
        try:
            result, row["decode_ms"] = timed(receive, rx, sent.prefix_boundaries, prefix, render_incrementally)
        except MissingReference:
            row["reference_rejected"] = True
            rows.append(row)
            continue
        if result.stage:
            output, synthesis_ms = timed(result.reconstruct)
            row["decode_ms"] += synthesis_ms
            row["display_calls"] = result.display_reconstructions
            row["entropy_records"] = result.entropy_decodes
            if quality:
                row["quality"] = quality.score(source.rgb[unit.start:unit.start + unit.valid], output)
        row["reference_committed"] = result.reference_committed
        if result.reference_committed:
            rx_hash = rx.adapter.reference.digest()
            if rx_hash != tx.adapter.reference.digest():
                raise ValueError(f"PHT R1 reference drift at {chunk_id}")
            row["reference_hash"] = rx_hash
        if pattern == "late_enhancement" and p_index == 4:
            row["late_discarded_bytes"] = late_probe(rx, result, sent.data[len(prefix):])
        rows.append(row)
    p_chunks = sum(not row["intra"] for row in rows)
    if p_chunks < 16:
        raise ValueError("the closed-loop run requires 16 P chunks")
    return {"mode": mode.name, "k_ref": k_ref, "qp": qp, "pattern": pattern, "p_chunks": p_chunks,
            "source_frames": len(source.yuv), "source_fps": source.fps, "quality": result_quality(rows),
            "physical_bytes": sum(row["bytes"] for row in rows), "per_unit": rows,
            "process_cpu_seconds": process_time() - start_cpu}


def receive(rx, boundaries, prefix, incremental):
    session = rx.begin_chunk()
    if not incremental:
        session.append_bytes(prefix)
        session.decode_new_layers()
        return session
    previous = 0
    for end in boundaries:
        if end > len(prefix):
            break
        session.append_bytes(prefix[previous:end])
        session.decode_new_layers()
        session.reconstruct()
        previous = end
    return session


def closed_study(source, qps, stock, tx, rx, quality, work):

    qp = qps[0]
    rows = []
    for mode in (Mode.MASK, Mode.PRECISION):
        for k_ref in (1, 2, 3):
            result = run_closed(source, qp, tx, rx, mode, k_ref, "base", quality)
            rows.append(result)
            print("closed base", mode.name, k_ref, flush=True)
    mode = Mode.MASK
    k_ref = 2
    for pattern in PATTERNS[1:]:
        score = quality if pattern in ("mid", "full", "alternating") else None
        rows.append(run_closed(source, qp, tx, rx, mode, k_ref, pattern, score))
        print("closed", pattern, flush=True)
    write_json(work / "reference_results.json", {"status": "PASS", "source": source.name,
               "runs": rows, "mode_under_loss": mode.name, "reference_prefix_under_loss": k_ref,
               "test_pixel_reads": 0, "validation_pixel_reads": 0})
    reference = official_profile(source, stock, qp)
    candidate = {}
    for pattern in ("base", "mid", "full", "incremental"):
        repetitions = []
        for _ in range(3):
            outcome = run_closed(source, qp, tx, rx, mode, k_ref,
                                 "full" if pattern == "incremental" else pattern,
                                 render_incrementally=pattern == "incremental")
            repetitions.append(profile_report(outcome["per_unit"], source, outcome["process_cpu_seconds"]))
        candidate[pattern] = repetitions
        print("profile", pattern, flush=True)
    write_json(work / "profiling.json", {"official": reference, "mask_kref2": candidate,
               "sampling": "native 1080p UVG at source 120 FPS, 16 P chunks, P-only steady state",
               "additional_capture_wait_at_30_fps_ms": 7000 / 30})
