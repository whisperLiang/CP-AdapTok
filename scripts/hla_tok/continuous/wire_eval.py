"""Physical wire evaluation, matched payload caps, and fixed-rate deadlines."""

from pathlib import Path

import torch
from torch import Tensor

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.evaluation.reconstruction import save_native
from hla_tok.streaming.continuous import MAX_K, NATIVE, TOKEN_DIM, ContinuousReceiver, RankOrder
from hla_tok.streaming.continuous_wire import (
    HEADER_BYTES,
    WireDtype,
    deserialize_continuous_tokens,
    max_budget_for_bytes,
    serialize_continuous_tokens,
    wire_rate,
)
from hla_tok.streaming.fixed13 import pack_tokens_13bit
from scripts.hla_tok.continuous.inputs import METRICS, Inputs, load_model, verify_model_state
from scripts.hla_tok.continuous.matrix import (
    Decode,
    Setting,
    compute,
    evaluate_matrix,
    groups,
    validation_frames,
)
from scripts.hla_tok.continuous.ordering import load_orders
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.continuous.receiver import compute_values
from scripts.hla_tok.v2.protocol import ROOT, csv_at, json_at

RATE_KEYS = ("order", "budget", "wire_dtype", "compute_dtype", "history")


def build_wire_decoder(inputs: Inputs) -> Decode:
    receiver = ContinuousReceiver.from_model(inputs.model)
    precision = inputs.config["quality_compute_dtype"]

    def decode(setting: Setting, values: Tensor, frame_id: int) -> tuple[Tensor, dict]:
        fmt = WireDtype(setting.wire_dtype)
        raw = serialize_continuous_tokens(values, fmt, frame_id, order=setting.order.name)
        packet = deserialize_continuous_tokens(raw)
        received = compute_values(packet.values.cuda(), precision)
        return receiver.decode(received, setting.order), {
            "payload_bytes": len(raw) - HEADER_BYTES, "header_bytes": HEADER_BYTES,
            "side_information_bytes": 0, "total_bytes": len(raw),
        }

    return decode


def summarize_wire(rows: list[dict]) -> list[dict]:
    report = []
    for fields, frames in groups(rows, RATE_KEYS):
        total = sum(frame["total_bytes"] for frame in frames)
        payload = sum(frame["payload_bytes"] for frame in frames)
        rate = wire_rate(payload, total, len(frames))
        report.append(fields | summarize(frames) | rate | {
            "actual_tokens_per_frame": fields["budget"], "time_span_frames": 1, "token_dim": TOKEN_DIM,
            "min_bytes_per_frame": min(frame["total_bytes"] for frame in frames),
            "max_bytes_per_frame": max(frame["total_bytes"] for frame in frames),
            "packet_rate_basis": "actual_bytes_one_packet_per_frame",
        })
    return report


def _settings(orders: list[RankOrder], budgets: list[int], formats: list[WireDtype]) -> list[Setting]:
    return [Setting(order, count, dtype.value) for order in orders for dtype in formats for count in budgets]


def _delta_from_raw(output: Path, rows: list[dict]) -> None:
    paths = (output / "fixed_budget.csv", output / "ordered_fixed_budget.csv")
    raw = {(row["order"], int(row["budget"])): row for path in paths for row in csv_at(path)}
    for row in rows:
        reference = raw[row["order"], row["budget"]]
        for metric in METRICS:
            row[f"wire_minus_raw_{metric}"] = row[metric] - float(reference[metric])


def _c1_feedback(v3: Path) -> dict:
    path = v3 / "results/hla_tok/v3/quantized/seed1/C1/full/step_10000/metrics.json"
    row, = (value for value in json_at(path) if value["mode"] == "fixed13")
    if row["frames"] != 174 or row["budget"] != MAX_K or row["rate_basis"] != "actual_fixed13_payload":
        raise ValueError("C1 reference is outside the matched Fixed13 endpoint")
    frame_bytes, remainder = divmod(row["total_bytes"], row["frames"])
    if remainder or frame_bytes != 26520:
        raise ValueError("C1 reference physical payload differs from the frozen endpoint")
    return {metric: row[metric] for metric in METRICS} | {
        "frames": 174, "method": "C1 Fixed13", "budget": MAX_K, "compute_dtype": "bf16",
        "wire_dtype": "fixed13", "payload_bytes_per_frame": frame_bytes, "payload_bpp": row["actual_bpp"],
        "source_metrics_sha256": sha256_file(path), "history": "stateless", "rate_basis": row["rate_basis"],
    }


def _curve(inputs: Inputs, quality: NativeQuality, output: Path, orders, formats, decode) -> list[dict]:
    settings = _settings(orders, inputs.config["budgets"], formats)
    frames = evaluate_matrix(inputs, quality, settings, output, "wire_formal", decode)
    report = summarize_wire(frames)
    _delta_from_raw(output, report)
    write_csv(output / "wire_rate_distortion.csv", report)
    write_csv(output / "wire_rate_distortion_per_frame.csv", frames)
    return report


def _equal_output(raw: list[dict], matched: list[dict], c1: dict) -> list[dict]:
    rows = []
    for dimension, source in (
        ("equal_token", [row for row in raw if row["budget"] == MAX_K]),
        ("equal_payload_cap", matched),
    ):
        rows.append(c1 | {"comparison": dimension, "comparability": "PASS"})
        for row in source:
            delta = {f"c0_minus_c1_{metric}": row[metric] - c1[metric] for metric in METRICS}
            rows.append({"method": "C0 continuous", "comparison": dimension, "comparability": "PASS"} | row | delta)
    rows.append({"method": "C1 Fixed13", "comparison": "equal_total_wire", "comparability": "NOT_COMPARABLE",
                 "reason": "The V3 C1 archive records the Fixed13 payload and has no full wire header ledger."})
    rows.append({"method": "Original AdapTok", "comparison": "equal_payload_cap", "comparability": "NOT_COMPARABLE",
                 "reason": "The matched archive codes native RGB in 16-frame windows. C0 codes independent frames."})
    return rows


def _equal_rate(inputs: Inputs, quality: NativeQuality, output: Path, orders, formats, decode,
                raw: list[dict], c1: dict) -> None:
    cap = c1["payload_bytes_per_frame"]
    counts = {dtype: max_budget_for_bytes(cap, dtype, include_header=False) for dtype in formats}
    settings = [Setting(order, counts[dtype], dtype.value) for order in orders for dtype in formats]
    frames = evaluate_matrix(inputs, quality, settings, output, "wire_equal_payload", decode)
    matched = summarize_wire(frames)
    for row in matched:
        remaining = cap - row["payload_bytes_per_frame"]
        width = WireDtype(row["wire_dtype"]).bytes_per_token
        if not 0 <= remaining < width:
            raise ValueError("matched C1 payload budget is not the largest legal continuous prefix")
        row["unused_payload_cap_bytes"] = remaining
        row["c1_payload_cap_bytes_per_frame"] = cap
    write_csv(output / "equal_rate_comparison.csv", _equal_output(raw, matched, c1))
    write_csv(output / "equal_rate_comparison_per_frame.csv", frames)


def _bandwidth_plan(inputs: Inputs, formats: list[WireDtype]) -> list[dict]:
    fps = inputs.config["target_fps"]
    return [{"target_mbps": mbps, "target_fps": fps, "wire_dtype": dtype.value,
             "max_bytes_per_deadline": cap, "budget": max_budget_for_bytes(cap, dtype)}
            for mbps in inputs.config["target_bandwidth_mbps"]
            for cap in (mbps * 1_000_000 // (fps * 8),) for dtype in formats]


def _deadline_summary(plan: list[dict], orders: list[RankOrder], frames: list[dict]) -> list[dict]:
    summary = {(row["order"], row["wire_dtype"], row["budget"]): row for row in summarize_wire(frames)}
    report = []
    for row in plan:
        for order in orders:
            name = order.name.name
            key = (name, row["wire_dtype"], row["budget"])
            if not row["budget"]:
                report.append(row | {"order": name, "rate_feasibility": "INFEASIBLE"})
                continue
            measured = summary[key]
            cap = row["max_bytes_per_deadline"]
            width = WireDtype(row["wire_dtype"]).bytes_per_token
            maximum = measured["max_bytes_per_frame"]
            if maximum > cap or (row["budget"] < MAX_K and maximum + width <= cap):
                raise ValueError("target did not take the largest legal budget by actual bytes")
            report.append(row | measured | {"rate_feasibility": "FEASIBLE", "overflow_frame_fraction": 0.0,
                                             "unused_bytes_per_deadline": cap - maximum})
    return report


def _bandwidth(inputs: Inputs, quality: NativeQuality, output: Path, orders, formats, decode) -> None:
    plan = _bandwidth_plan(inputs, formats)
    settings = list(dict.fromkeys(
        Setting(order, row["budget"], row["wire_dtype"]) for row in plan if row["budget"] for order in orders
    ))
    frames = evaluate_matrix(inputs, quality, settings, output, "wire_bandwidth", decode) if settings else []
    write_csv(output / "bandwidth_constrained_quality.csv", _deadline_summary(plan, orders, frames))
    if frames:
        write_csv(output / "bandwidth_constrained_quality_per_frame.csv", frames)


def archive_original(output: Path, inputs: Inputs) -> None:
    baseline = ROOT / "results/hla_tok/baselines"
    weights = json_at(baseline / "official_weights.json")
    if weights["metrics"] != inputs.metric_weights:
        raise ValueError("archived Original AdapTok metric weights disagree")
    rows = [row for row in csv_at(baseline / "metrics.csv") if row["method"] == "Original AdapTok"
            and row["frames"] == "174" and row["precision"] == "bf16" and row["stride"] == "128"]
    frames = csv_at(baseline / "stride128_bf16/per_frame.csv")
    expected = {(video, index) for video, count in inputs.cohort.items() for index in range(count)}
    if {(row["video"], int(row["frame"])) for row in frames} != expected:
        raise ValueError("archived Original AdapTok frames differ")
    report = [row | {"comparison_status": "NOT_COMPARABLE", "history": "16_frame_input_windows",
                     "cohort_sha256": inputs.config["cohort_sha256"],
                     "rate_scope": "actual Fixed13 payload, full header unavailable"} for row in rows]
    write_csv(output / "original_adaptok_context.csv", report)


@torch.inference_mode()
def _fixed13_sample(inputs: Inputs, v3: Path, output: Path, c1: dict) -> None:
    model, manifest = load_model(v3, "C1", inputs.config)
    sample = inputs.config["sample"]
    video, frame, source = next(validation_frames(inputs, [(sample["video"], sample["frame"])]))
    with compute("bf16"):
        encoded = model.encode_training(source, MAX_K, quantize=True)
        if encoded.ids is None:
            raise ValueError("C1 Fixed13 sample requires physical token IDs")
        payload = pack_tokens_13bit(encoded.ids[0].cpu().tolist())
        image = model.receiver().decode_bytes(payload, MAX_K, NATIVE)
    if len(payload) != c1["payload_bytes_per_frame"]:
        raise ValueError("C1 sample did not use the real matched Fixed13 payload")
    save_native(output / "samples", "C1_fixed13_K16320", image)
    write_json(output / "samples/c1_sample.json", {"video": video, "frame": frame, "bytes": len(payload),
                                                 "checkpoint_sha256": manifest["sha256"],
                                                 "state_dict_sha256": verify_model_state(model, manifest["state_dict_sha256"])})


@torch.inference_mode()
def run_wire_evaluation(inputs: Inputs, v3: Path, output: Path) -> None:
    decision = json_at(output / "ordering_decision.json")
    orders = [order for order in load_orders(output) if order.name.name in decision["wire_orders"]]
    formats = [WireDtype(item) for item in inputs.config["wire_formats"]]
    quality = NativeQuality()
    decode = build_wire_decoder(inputs)
    raw = _curve(inputs, quality, output, orders, formats, decode)
    c1 = _c1_feedback(v3)
    _equal_rate(inputs, quality, output, orders, formats, decode, raw, c1)
    archive_original(output, inputs)
    selected = [order for order in orders if order.name.name in {"O0", decision["primary_order"]}]
    _bandwidth(inputs, quality, output, selected, formats, decode)
    _fixed13_sample(inputs, v3, output, c1)
