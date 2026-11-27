"""Trace replay with actual HT-S and PHT payloads and causal QP decisions."""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch

from baselines.dcvc_uf.native import NativeNetworks
from baselines.dcvc_uf.official import NativeNAL
from baselines.dcvc_uf.units import NativeSchedule
from benchmarks.pht_tok.closed import official_encode
from benchmarks.pht_tok.study import ROOT, SKIP, Quality, uvg_source
from common.artifacts import write_json
from common.performance import timed
from pht_tok.bitstream import Mode
from pht_tok.codec import MissingReference, PHTDecoder, PHTEncoder, model_identifier
from pht_tok.streaming import (
    PACKET_OVERHEAD,
    Observation,
    RateTrace,
    Transport,
    goodput_at,
    network_sizes,
    select_prefix,
)
from third_party.DCVC.src.utils.common import set_torch_env

STATIC = (0.1, 0.25, 0.5, 1, 2, 5, 10, 20, 40)
BUFFER = 0.5
SAFETY = 0.020
REFERENCE_PREFIX = 2
QP_HISTORICAL_PRIOR = 32
INITIAL_BPS = 3e6
PACING_FRACTION = 0.9


@dataclass(frozen=True, slots=True)
class Scenario:
    name: str
    trace: RateTrace
    rtt: float
    jitter: float = 0.0
    loss: float = 0.0


def cases(verizon: Path | None) -> list[Scenario]:
    def scenario(name, values, rtt, jitter=0.0, loss=0.0):
        return Scenario(name, RateTrace(tuple(values)), rtt, jitter, loss)
    rng = np.random.default_rng(272026)
    values = rng.uniform(0.4, 12, 12)
    scenarios = [
        scenario("high_low", ((0, 12), (0.27, 0.65), (0.72, 13), (1.01, 0.9), (1.38, 10)), 0.04),
        scenario("burst", ((0, 14), (0.38, 0.4), (0.56, 14), (0.81, 0.6), (0.97, 13)), 0.10, 0.008),
        scenario("sawtooth", tuple((i * 0.12, 1 + 2.5 * (i % 5)) for i in range(12)), 0.06, 0.008),
        scenario("random", tuple((i * 0.12, rate) for i, rate in enumerate(values)), 0.07, 0.012, 0.003),
    ]
    if verizon:
        times = np.loadtxt(verizon, dtype=int)
        interval_ms = 100
        bins = np.arange(times[0], times[-1] + interval_ms, interval_ms)
        counts, _ = np.histogram(times, bins)
        # Mahimahi schedules one 1500-byte delivery per trace line.
        rates = np.maximum(0.001, counts * 1500 * 8 / (interval_ms / 1000) / 1e6)
        scenarios.append(scenario("Verizon_LTE_short_100ms",
                                  tuple((i * interval_ms / 1000, rate) for i, rate in enumerate(rates)),
                                  0.06, 0.006))
    return scenarios + [scenario(f"static_{rate:g}_Mbps", ((0, rate),), 0.04) for rate in STATIC]


def rate_models(fixed: dict) -> dict:
    unit_rows = fixed["units"]
    models = {}
    for method, k_ref in (("official", 4), ("MASK", REFERENCE_PREFIX)):
        points = {}
        for intra in (False, True):
            rows = [row for row in unit_rows if (row["method"], row["stage"], row["intra"]) == (
                method, k_ref, intra)]
            qps = sorted({row["qp"] for row in rows})
            typical = [mean(network_sizes((row["bytes"],))[0] for row in rows if row["qp"] == qp) for qp in qps]
            predicted = np.exp(np.interp(np.arange(64), qps, np.log(typical)))
            points[intra] = np.maximum.accumulate(predicted)
        models[method] = points
    return models


class WireCodec:
    def __init__(self, weights: Path):
        self.tx_stock = NativeNetworks(weights, "hts", SKIP)
        self.rx_stock = NativeNetworks(weights, "hts", SKIP)
        model_id = model_identifier(weights, SKIP)
        self.tx = PHTEncoder(NativeNetworks(weights, "hts", SKIP), model_id)
        self.rx = PHTDecoder(NativeNetworks(weights, "hts", SKIP), model_id)
        from inference_extensions_cuda import DMCIProxy
        intra = self.rx_stock.intra
        intra.proxy = DMCIProxy()
        intra.proxy.set_param(intra.add_cdf_to_state_dict(intra.state_dict()), SKIP)
        self.reset()

    def reset(self):
        self.tx.key = self.rx.key = None
        self.writer = self.reader = NativeNAL()
        self.last = None

    def encode(self, progressive, source, unit, qp):
        if not progressive:
            if unit.intra:
                self.writer = NativeNAL()
            raw = official_encode(self.tx_stock, source, unit, qp, self.writer)
            return raw, (len(raw),)
        frames = source.yuv[unit.start:unit.start + unit.valid]
        stream = (self.tx.encode_intra(frames, qp, unit.start) if unit.intra else
                  self.tx.encode_chunk_once(frames, qp, unit.start, REFERENCE_PREFIX,
                                            reset=unit.reset, mode=Mode.MASK))
        return stream.data, stream.prefix_boundaries

    def reconstruct(self, progressive, raw, unit, last_unit):
        if progressive:
            session = self.rx.decode_prefix(raw)
            return session.reconstruct(), session.reference_committed
        if not unit.intra and self.last != last_unit:
            raise MissingReference("whole native HT-S reference is missing")
        packet = self.reader.decode(raw, unit.start)
        result = self.rx_stock.decode(packet)
        self.last = unit.start
        return result[:unit.valid], True


def choose_qp(predicted, factor: float, goodput: float, seconds_left: float) -> int:
    available_bytes = max(0, seconds_left) * goodput / 8
    return max(0, int(np.searchsorted(predicted * factor, available_bytes, side="right")) - 1)


def selected_stage(policy, boundaries, estimated, remaining, reserve, minimum, transport, ready, deadline):
    if policy == "S3_fixed_prefix":
        return minimum
    if policy == "S6_oracle":
        # Offline bound for prefix selection at the same causal pre-encode QP.
        for i in range(len(boundaries), minimum - 1, -1):
            done = transport.trace.finish(max(ready, transport.free), network_sizes(boundaries)[i - 1])
            if done + transport.rtt / 2 + reserve <= deadline:
                return i
        return minimum
    selected = select_prefix(boundaries, estimated, remaining, 0.0, reserve, SAFETY, minimum=minimum)
    if policy == "S2_whole_gated":
        return selected
    return selected or minimum


def measured_times(profiles, progressive):
    if progressive:
        variants = profiles["mask_kref2"]
        stages = {stage: variants[key] for stage, key in ((2, "base"), (3, "mid"), (4, "full"))}
        primary = variants["full"]
    else:
        primary = profiles["official_repetitions"]
        stages = {1: primary}
    return {
        "p_encode": median(row["encode_ms_per_chunk_p50_p95_p99"][0] for row in primary) / 1000,
        "p_reserve": median(row["decode_ms_per_chunk_p50_p95_p99"][1] for row in primary) / 1000,
        "p_decode": {stage: median(row["decode_ms_per_chunk_p50_p95_p99"][0] for row in rows) / 1000
                     for stage, rows in stages.items()},
        "i_encode": median(row["per_unit"][0]["encode_ms"] for row in primary) / 1000,
        "i_decode": median(row["per_unit"][0]["decode_ms"] for row in primary) / 1000,
    }


def simulate(codec, source, policy, scenario, models, quality, profiles) -> dict:
    codec.reset()
    progressive = policy in ("S3_fixed_prefix", "S4_adaptive_prefix", "S6_oracle")
    adaptive_qp = policy not in ("S0_fixed_qp", "S3_fixed_prefix")
    control = models["MASK" if progressive else "official"]
    timings = measured_times(profiles, progressive)
    units = list(NativeSchedule(8, gop_frames=65, reset_interval=128).units(len(source.yuv)))
    network = Transport(scenario.trace, seed=272026, rtt_seconds=scenario.rtt,
                        jitter_seconds=scenario.jitter, packet_loss=scenario.loss)
    observations: list[Observation] = []
    rows = []
    factor = 1.0
    local_encoder_free = local_decoder_free = estimated_link_free = 0.0
    previous_capture = None
    for unit in units:
        capture_time = (unit.start + unit.valid - 1) / source.fps
        deadline = unit.start / source.fps + BUFFER
        encoder_start = max(capture_time, local_encoder_free)
        encode_reserve = timings["i_encode"] if unit.intra else timings["p_encode"]
        decode_reserve = timings["i_decode"] if unit.intra else timings["p_reserve"]
        estimated = goodput_at(tuple(observations), encoder_start, INITIAL_BPS)
        if observations and observations[-1].at <= encoder_start:
            estimated_link_free = encoder_start
        source_interval = 8 / source.fps
        pacing = PACING_FRACTION * source_interval
        queue_ready = max(encoder_start + encode_reserve, estimated_link_free)
        headroom = deadline - queue_ready - decode_reserve - scenario.rtt / 2 - SAFETY
        available = min(pacing, headroom)
        qp = choose_qp(control[unit.intra], factor, estimated, available) if adaptive_qp else QP_HISTORICAL_PRIOR
        (raw, boundaries), encode_ms = timed(codec.encode, progressive, source, unit, qp)
        ready = local_encoder_free = encoder_start + encode_reserve
        minimum = REFERENCE_PREFIX if progressive and not unit.intra else 1
        wire_sizes = network_sizes(boundaries)
        factor = 0.65 * factor + 0.35 * (wire_sizes[minimum - 1] / control[unit.intra][qp])
        remaining = min(pacing + decode_reserve + SAFETY,
                        deadline - max(ready, estimated_link_free) - scenario.rtt / 2)
        stage = (len(boundaries) if policy in ("S0_fixed_qp", "S1_adaptive_qp") else selected_stage(
            policy, boundaries, estimated, remaining, decode_reserve, minimum, network, ready, deadline))
        row = {"source_start": unit.start, "valid_frames": unit.valid, "intra": unit.intra, "qp": qp,
               "goodput_estimate_bps": estimated, "deadline": deadline, "encoded_at": ready,
               "causal_pacing_seconds": pacing, "attempted_stage": stage, "received_stage": 0, "displayed_stage": 0, "reference_commit": False,
               "actual_sent_bytes": 0, "actual_received_bytes": 0, "useful_delivered_bytes": 0,
               "codec_sent_bytes": 0, "forward_headers_sent_bytes": 0, "forward_headers_received_bytes": 0,
               "late_received_bytes": 0, "late_enhancement_bytes": 0, "encode_ms": encode_ms, "decode_ms": 0.0,
               "scheduled_encode_ms": encode_reserve * 1000}
        if stage:
            estimated_link_free = max(ready, estimated_link_free) + wire_sizes[stage - 1] * 8 / max(estimated, 1)
            delivery = network.send(ready, boundaries[:stage])
            observations.append(delivery.observation)
            observe_deadline = deadline - decode_reserve
            available_stage = (
                delivery.stage_by(deadline, decode_reserve) if local_decoder_free <= observe_deadline else 0
            )
            feedback_bytes = 48 if delivery.bytes_received else 0
            row.update(actual_sent_bytes=delivery.bytes_sent, actual_received_bytes=delivery.bytes_received,
                       feedback_wire_bytes=feedback_bytes, codec_sent_bytes=boundaries[stage - 1],
                       forward_headers_sent_bytes=len(delivery.packets) * PACKET_OVERHEAD,
                       forward_headers_received_bytes=sum(
                           PACKET_OVERHEAD for datagram in delivery.packets if datagram.arrival is not None
                       ),
                       late_received_bytes=delivery.bytes_received - delivery.received_by(deadline),
                       received_stage=available_stage,
                       late_enhancement_bytes=max(0, delivery.bytes_received - max(
                           wire_sizes[minimum - 1], delivery.received_by(deadline)
                       )) if progressive else 0)
            if available_stage >= minimum:
                boundary = boundaries[available_stage - 1]
                arrival = delivery.arrivals[available_stage - 1]
                assert arrival is not None
                try:
                    (decoded, committed), decode_ms = timed(
                        codec.reconstruct, progressive, raw[:boundary], unit, previous_capture)
                except MissingReference:
                    row["reference_rejected"] = True
                else:
                    scheduled = timings["i_decode"] if unit.intra else timings["p_decode"][available_stage]
                    local_decoder_free = max(arrival, local_decoder_free) + scheduled
                    row.update(decode_ms=decode_ms, scheduled_decode_ms=scheduled * 1000,
                               reference_commit=committed)
                    if local_decoder_free <= deadline:
                        row.update(displayed_stage=available_stage, useful_delivered_bytes=boundary,
                                   capture_to_display_ms=(deadline - unit.start / source.fps) * 1000,
                                   receiver_ready_ms=(local_decoder_free - unit.start / source.fps) * 1000)
                        row["quality"] = quality.score(source.rgb[unit.start:unit.start + unit.valid], decoded)
        rows.append(row)
        previous_capture = unit.start
    return playback_summary(rows, policy, scenario)


def playback_summary(rows, policy, scenario):
    quality = [frame for row in rows for frame in row.get("quality", [])]
    count = sum(row["valid_frames"] for row in rows)
    showed = len(quality)
    sums = {field: sum(row[field] for row in rows) for field in (
        "actual_sent_bytes", "actual_received_bytes", "late_received_bytes", "late_enhancement_bytes",
        "useful_delivered_bytes", "codec_sent_bytes", "forward_headers_sent_bytes",
        "forward_headers_received_bytes")}
    if sums["actual_sent_bytes"] != sums["codec_sent_bytes"] + sums["forward_headers_sent_bytes"]:
        raise ValueError("physical network and codec byte ledgers disagree")
    metrics = {name: mean(row[name] for row in quality) for name in quality[0]} if quality else {}
    fluctuation = mean(abs(b["psnr"] - a["psnr"]) for a, b in zip(quality, quality[1:])) if showed > 1 else None
    return {"trace": scenario.name, "policy": policy, "rtt_ms": 1000 * scenario.rtt,
            "jitter_ms": 1000 * scenario.jitter, "configured_packet_loss": scenario.loss,
            "frames": count, "displayed_frames": showed, "dropped_frames": count - showed,
            "deadline_hit_rate": showed / count, "stall_seconds": 0.0,
            "startup_seconds": next((r["deadline"] for r in rows if r["displayed_stage"]), None),
            "quality_fluctuation_psnr_db": fluctuation, "metrics": metrics,
            "qoe_drop_penalized_psnr": (sum(frame["psnr"] for frame in quality) - 20 * (count - showed)) / count,
            "wasted_bytes_including_transport": sums["actual_sent_bytes"] - sums["useful_delivered_bytes"],
            "feedback_wire_bytes": sum(r.get("feedback_wire_bytes", 0) for r in rows),
            "recovery_i_wire_bytes": sum(r["actual_sent_bytes"] for r in rows[1:] if r["intra"]),
            **sums, "units": rows}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--uvg", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=ROOT / ".build/pht-tok")
    parser.add_argument("--verizon", type=Path, default=ROOT / "data/traces/verizon-lte.down")
    parser.add_argument("--calibration", type=Path, default=ROOT / "configs/pht_tok/network_calibration.json")
    parser.add_argument("--profile", type=Path, default=ROOT / "results/pht_tok/profiling.json")
    args = parser.parse_args()
    set_torch_env()
    rates = rate_models(json.loads(args.calibration.read_text()))
    profile = json.loads(args.profile.read_text())
    policies = ("S0_fixed_qp", "S1_adaptive_qp", "S2_whole_gated", "S3_fixed_prefix",
                "S4_adaptive_prefix", "S6_oracle")
    with torch.cuda.stream(torch.cuda.Stream()):
        codec = WireCodec(args.weights)
        source = uvg_source(args.uvg, 129)
        quality = Quality()
        fast_metrics = Quality(False)
        rows = []
        for case in cases(args.verizon):
            static = case.name.startswith("static_")
            selected = ("S1_adaptive_qp", "S4_adaptive_prefix") if static else policies
            for policy in selected:
                destination = args.work / f"network-{case.name}-{policy}.json"
                result = simulate(codec, source, policy, case, rates, fast_metrics if static else quality, profile)
                write_json(destination, result)
                rows.append(result)
                print(case.name, policy, result["deadline_hit_rate"], flush=True)
    write_json(args.work / "network_results.json",
               {"runs": rows, "source": source.name, "source_fps": source.fps,
                "playout_buffer_ms": BUFFER * 1000, "safety_margin_ms": SAFETY * 1000,
                "overhead": "40 B per forward IPv4/UDP/RTP packet; 48 B per modeled bandwidth observation in reverse",
                "refresh": "independent I-frame after every eight P-chunks; drop and resume without stall",
                "pacing": "both adaptive codecs cap new per-chunk service time at 90% of acquisition interval",
                "oracle": "uses actual future service at current QP per chunk. QoE is measured on its own path",
                "runtime_model": "frozen H100 steady-state medians per codec and stage, same warmup protocol; "
                                 "per-unit measured wall latencies also recorded",
                "test_pixel_reads": 0, "validation_pixel_reads": 0})


if __name__ == "__main__":
    main()
