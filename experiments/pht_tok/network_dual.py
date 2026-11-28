"""Packet-level replay of real native and PHT2 streams under matched deadlines."""

import csv
from functools import partial
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from common.artifacts import write_csv, write_json
from experiments.pht_tok.evaluate import SPLIT, Case, Endpoints, clone_frames, exact_tuple, validation_sources
from experiments.pht_tok.study import Quality, calibration
from pht_tok.streaming import RateTrace, Transport, goodput_at, network_sizes, select_prefix

QPS = (16, 32, 48, 63)
PLAYOUT = 0.5
RTT = 0.04
FEEDBACK_BYTES = 48
INITIAL_BPS = 3e6


def traces():
    jitter = np.random.default_rng(272026).uniform(0.25, 8.0, 25)
    return {
        "stable": RateTrace(((0.0, 2.5),)),
        "pre_encode_change": RateTrace(((0.0, 7.0), (0.035, 0.8), (0.65, 5.0))),
        "post_encode_drop": RateTrace(((0.0, 7.0), (0.28, 0.35), (0.57, 7.0))),
        "recovery": RateTrace(((0.0, 0.4), (0.52, 9.0), (1.25, 2.4))),
        "random": RateTrace(tuple((i * 0.09, float(rate)) for i, rate in enumerate(jitter))),
    }


def calibration_sizes(weights, sources, backend):
    ep = Endpoints(weights, Case(backend))
    result = {}
    for qp in QPS:
        sizes = {True: [], False: []}
        for source in sources:
            ep.reset()
            for unit in ep.case.units(len(source.rgb)):
                raw = ep.encode_native(source, unit, qp)
                sizes[unit.intra].append(network_sizes((len(raw),))[0])
        result[qp] = {kind: mean(values) for kind, values in sizes.items()}
    return result


def choose_qp(models, unit, budget):
    return max((qp for qp, counts in models.items() if counts[unit.intra] <= budget), default=min(models))


def display_score(frame_scores, done, source, unit):
    ready = [(unit.start + i) / source.fps + PLAYOUT for i in range(unit.valid)]
    shown = [row["psnr"] for row, deadline in zip(frame_scores, ready, strict=True) if done <= deadline]
    return len(shown), sum(shown) - 20 * (unit.valid - len(shown))


def eligible(trace, starts, wire_sizes, decode_free, decode, deadline):
    return tuple(i for i, size in enumerate(wire_sizes, 1)
                 if max(trace.finish(starts, size) + RTT / 2, decode_free) + decode <= deadline)


def oracle_prefix(ep, packet, source, unit, trace, link_ready, decoder_ready, dec_seconds, quality):
    if unit.intra:
        return 1
    first_deadline = unit.start / source.fps + PLAYOUT
    stages = eligible(trace, link_ready, network_sizes(packet.prefix_boundaries),
                      decoder_ready, dec_seconds, first_deadline)
    if not stages:
        return 1
    original = source.rgb[unit.start:unit.start + unit.valid]
    values = []
    for stage in stages:
        prefix = packet.prefix(max_bytes=packet.prefix_boundaries[stage - 1])
        frames = ep.rx.decode_prefix(prefix, commit=False).reconstruct()
        values.append(mean(row["psnr"] for row in quality.score(original, frames)))
    return max(zip(values, stages, strict=True), key=lambda point: (point[0], -point[1]))[1]


def session_replay(ep, source, trace, timing, models, quality, oracle=False):
    ep.reset()
    transport = Transport(trace, seed=272026, rtt_seconds=RTT)
    enc_seconds, dec_seconds = timing
    observations = []
    rows = []
    enc_free = dec_free = forecast_free = 0.0
    for unit in ep.case.units(len(source.rgb)):
        acquired = (unit.start + unit.valid - 1) / source.fps
        start = max(acquired, enc_free)
        deadline = unit.start / source.fps + PLAYOUT
        estimate = goodput_at(tuple(observations), start, INITIAL_BPS)
        predicted_start = max(start, forecast_free)
        budget = max(0.0, min(deadline - predicted_start - enc_seconds - dec_seconds - RTT / 2,
                              unit.coded / source.fps)) * estimate / 8
        qp = choose_qp(models, unit, budget)
        enc_free = start + enc_seconds
        if ep.case.mode is None:
            wire = ep.encode_native(source, unit, qp)
            boundaries = (len(wire),)
        else:
            packet = ep.encode_progressive(source, unit, qp)
            boundaries = packet.prefix_boundaries
            if oracle:
                stage = oracle_prefix(ep, packet, source, unit, trace, max(enc_free, transport.free),
                                      dec_free, dec_seconds, quality)
            else:
                stage = max(1, select_prefix(boundaries, estimate, deadline - max(enc_free, forecast_free),
                                             0.0, dec_seconds, RTT / 2))
            boundaries = boundaries[:stage]
            wire = packet.prefix(max_bytes=boundaries[-1])
        delivery = transport.send(enc_free, boundaries)
        # A complete reference is always sent. A late reference is decoded for future units.
        first_deadline = deadline
        timely_stage = (delivery.stage_by(first_deadline, dec_seconds)
                        if dec_free + dec_seconds <= first_deadline else 0)
        received = max(1, timely_stage) if ep.case.mode is not None else 1
        arrival = delivery.arrivals[received - 1]
        if arrival is None:
            raise ValueError("these trace cases require ordered lossless packet delivery")
        dec_free = max(arrival, dec_free) + dec_seconds
        if ep.case.mode is None:
            frames = clone_frames(ep.decode_native(unit, wire), unit, source)
        else:
            frames = ep.decode_progressive(wire[:boundaries[received - 1]])
            exact_tuple(ep.tx.adapter.reference.tensors, ep.rx.adapter.reference.tensors, "network reference state")
        original = source.rgb[unit.start:unit.start + unit.valid]
        shown, score = display_score(quality.score(original, frames), dec_free, source, unit)
        rows.append({"source": source.name, "method": ep.case.name + ("_oracle" if oracle else ""),
                     "start": unit.start, "frames": unit.valid, "qp": qp, "encoded_ms": enc_free * 1000,
                     "deadline_ms": deadline * 1000, "display_ms": dec_free * 1000,
                     "stage_sent": len(boundaries), "stage_decoded": received, "payload_bytes": len(wire),
                     "reference_bytes": boundaries[0], "wire_bytes": delivery.bytes_sent,
                     "late_wire_bytes": delivery.bytes_sent - delivery.received_by(deadline),
                     "feedback_bytes": FEEDBACK_BYTES, "on_time_frames": shown, "penalized_psnr_sum": score})
        observations.append(delivery.observation)
        forecast_free = max(enc_free, forecast_free) + delivery.bytes_sent * 8 / estimate
    return rows


def summarize(rows):
    frames = sum(r["frames"] for r in rows)
    shown = sum(r["on_time_frames"] for r in rows)
    return {"frames": frames, "on_time_frames": shown, "on_time_fraction": shown / frames,
            "drop_penalized_psnr_db": sum(r["penalized_psnr_sum"] for r in rows) / frames,
            **{key: sum(r[key] for r in rows) for key in
               ("payload_bytes", "wire_bytes", "feedback_bytes", "late_wire_bytes", "reference_bytes")}}


def video_intervals(rows):
    sources = sorted({r["source"] for r in rows})
    curves = {}
    for backend in SPLIT:
        values = []
        for source in sources:
            sample = [r for r in rows if r["source"] == source]
            a = summarize([r for r in sample if r["method"] == f"native_{backend}"])
            b = summarize([r for r in sample if r["method"] == f"pht2_{backend}_r1"])
            values.append(b["drop_penalized_psnr_db"] - a["drop_penalized_psnr_db"])
        draws = np.random.default_rng(272026).choice(values, (10000, len(values)), replace=True).mean(axis=1)
        curves[backend] = {"per_video_db": dict(zip(sources, values, strict=True)),
                           "bootstrap_95_db": np.percentile(draws, [2.5, 97.5]).tolist()}
    return curves


@torch.inference_mode()
def network(weights: Path, output: Path):
    with (output / "throughput.csv").open() as handle:
        profiles = {r["method"]: r for r in csv.DictReader(handle)}
    validate, reads = validation_sources()
    train, train_reads = calibration()
    quality = Quality(perceptual=False)
    endpoints = {case.name: Endpoints(weights, case)
                 for backend, mode in SPLIT.items() for case in (Case(backend), Case(backend, mode, 1))}
    models = {name: calibration_sizes(weights, train[:2], name) for name in SPLIT}
    details: list[dict] = []
    aggregate, intervals = [], {}
    for scenario, trace in traces().items():
        rows = []
        for method, ep in endpoints.items():
            timings = tuple(float(profiles[method][f"{side}_ms_p50"]) / 1000 for side in ("encode", "decode"))
            for oracle in (False, True) if ep.case.mode is not None else (False,):
                run = partial(session_replay, ep, trace=trace, timing=timings,
                              models=models[ep.case.backend], quality=quality, oracle=oracle)
                records = [row for source in validate for row in run(source)]
                rows.extend(records)
                aggregate.append({"trace": scenario, "method": records[0]["method"], **summarize(records)})
        details.extend({"trace": scenario, **row} for row in rows)
        intervals[scenario] = video_intervals(rows)
        print("Dynamic", scenario, flush=True)
    write_csv(output / "dynamic_network.csv", aggregate)
    write_json(output / "dynamic_evidence.json", {
        "reads": reads, "calibration_reads": train_reads, "calibration_clip_count": 2,
        "paired_video_intervals": intervals, "traces_mbps": {name: trace.knots for name, trace in traces().items()},
        "oracle": "Clairvoyant feasible best PSNR for the current unit at its first-frame deadline and actual queue; no global optimality claim.",
        "playout_s": PLAYOUT, "rtt_s": RTT, "feedback_bytes_per_unit": FEEDBACK_BYTES,
        "timing": "QP32 synchronized H100 medians per method, constant per method across traces",
        "qp": "Both paths choose QP before encoding with the same backend-specific Calibration model and available feedback.",
        "protocol": "ordered lossless IPv4/UDP/RTP accounting with 1200-byte application payload, one FIFO link",
        "units": details})
