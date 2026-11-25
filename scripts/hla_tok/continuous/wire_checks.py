"""Exercise the actual byte path with real C0 tensors in both compute precisions."""

import json
import os
import subprocess
import sys
from itertools import combinations
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from torch import Tensor

from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.streaming.continuous import MAX_K, ContinuousReceiver, RankOrder, encode_continuous, truncate
from hla_tok.streaming.continuous_wire import (
    HEADER_BYTES,
    FrameReceiver,
    WireDtype,
    deserialize_continuous_tokens,
    serialize_continuous_tokens,
    serialize_cpu,
    serialize_enhancement,
)
from scripts.hla_tok.continuous.contracts import prefix_plan
from scripts.hla_tok.continuous.inputs import Inputs
from scripts.hla_tok.continuous.matrix import compute, representative, validation_frames
from scripts.hla_tok.continuous.ordering import load_orders
from scripts.hla_tok.continuous.receiver import compute_values, tensor_digest
from scripts.hla_tok.v2.protocol import ROOT, json_at


def _bits(values: Tensor) -> Tensor:
    return values.contiguous().view(torch.uint8)


def _packet_audit(values: Tensor, dtype: WireDtype, order: RankOrder, frame: int, plan: dict) -> dict:
    packets = {count: serialize_continuous_tokens(truncate(values, count), dtype, frame, order=order.name)
               for count in plan["all"]}
    full = packets[MAX_K]
    expected_cpu = values.to(dtype.torch_dtype).cpu()
    for count, raw in packets.items():
        packet = deserialize_continuous_tokens(raw)
        if not torch.equal(_bits(packet.values), _bits(expected_cpu[:, :count])):
            raise ValueError(f"{dtype.value} round-trip changed bits at K{count}")
        if len(raw) != HEADER_BYTES + count * dtype.bytes_per_token or raw[HEADER_BYTES:] != full[HEADER_BYTES:len(raw)]:
            raise ValueError(f"{dtype.value} payload length or strict prefix failed at K{count}")
    for lower, upper in combinations(plan["formal"], 2):
        if packets[lower][HEADER_BYTES:] != packets[upper][HEADER_BYTES:len(packets[lower])]:
            raise ValueError("formal byte pair is not a strict prefix")
    return {"wire_dtype": dtype.value, "order": order.name.name,
            "payload_bytes_K16320": len(full) - HEADER_BYTES, "total_bytes_K16320": len(full),
            "prefixes": len(packets), "formal_pairs": 28, "roundtrip_standard_conversion_exact": True}


def _segments(values: Tensor, dtype: WireDtype, order: RankOrder, frame: int) -> list[bytes]:
    return [serialize_enhancement(values, dtype, frame, start, end, order.name)
            for start, end in ((0, 510), (510, 6120), (6120, MAX_K))]


def _committed_parts(segments: list[bytes], dtype: WireDtype, order: RankOrder, frame: int) -> Tensor:
    receive = FrameReceiver(frame, dtype, order.name)
    counts = [receive.accept(segments[2]), receive.accept(segments[0])]
    early = receive.prefix()
    counts.append(receive.accept(segments[1]))
    complete = receive.commit()
    if counts != [0, 510, MAX_K] or early is None or complete is None:
        raise ValueError("out-of-order enhancement did not preserve the contiguous prefix")
    if not torch.equal(early, complete[:, :510]):
        raise ValueError("enhancement rewrote existing tokens")
    return complete


def _incremental(values: Tensor, receiver: ContinuousReceiver, order: RankOrder, dtype: WireDtype,
                 frame: int, precision: str) -> dict:
    one = serialize_continuous_tokens(values, dtype, frame, order=order.name)
    parts = _segments(values, dtype, order, frame)
    complete = _committed_parts(parts, dtype, order, frame)
    single = deserialize_continuous_tokens(one).values
    first_rgb = receiver.decode(compute_values(single.cuda(), precision), order)
    combined_rgb = receiver.decode(compute_values(complete.cuda(), precision), order)
    no_resend = len(one) + (len(parts) - 1) * HEADER_BYTES == sum(map(len, parts))
    if not (no_resend and torch.equal(_bits(complete), _bits(single)) and torch.equal(first_rgb, combined_rgb)):
        raise ValueError("incremental payload or decoded output differs from one-shot")
    return {"order": order.name.name, "wire_dtype": dtype.value, "compute_dtype": precision,
            "payload_bytes": len(one) - HEADER_BYTES, "one_shot_total_bytes": len(one),
            "incremental_total_bytes": sum(map(len, parts)), "segments": len(parts),
            "no_duplicate_tokens": no_resend, "decode_rgb_bit_exact": True}


def _independent(receiver: ContinuousReceiver, order: RankOrder, values: Tensor, output: Path) -> dict:
    dtype, frame, precision = WireDtype.BF16, 217, "bf16"
    segments = _segments(values, dtype, order, frame)
    gpu = compute_values(_committed_parts(segments, dtype, order, frame).cuda(), precision)
    with compute(precision):
        rgb = receiver.decode(gpu, order)
    with TemporaryDirectory(prefix="hla-continuous-receiver-") as name:
        directory = Path(name)
        weights = directory / "synthesis.pt"
        torch.save(receiver.state_dict(), weights)
        filenames = [directory / f"fragment-{i}.hlac" for i in range(len(segments))]
        for path, packet in zip(filenames, segments, strict=True):
            path.write_bytes(packet)
        args = [sys.executable, "-m", "scripts.hla_tok.continuous.receiver", "--weights", str(weights),
                "--ordering", str(output), "--precision", precision, *(str(filenames[i]) for i in (2, 0, 1))]
        received = json.loads(subprocess.check_output(args, cwd=ROOT, text=True))
        received["synthesis_only_weights_sha256"] = sha256_file(weights)
    if received["pid"] == os.getpid() or received["rgb_sha256"] != tensor_digest(rgb):
        raise ValueError("independent synthesis did not reproduce the sender-side round-trip RGB")
    if received["received_tokens_sha256"] != tensor_digest(gpu) or received["modules"] != [
            "local_decoder", "rgb_decoder"]:
        raise ValueError("independent receiver used an unexpected tensor or synthesis architecture")
    return received | {"sender_pid": os.getpid(), "bytes_only_frame_input": True, "rgb_bit_exact": True,
                       "pre_shared_order_manifest_sha256": sha256_file(output / "rank_order_manifest.json")}


def _all_legal_prefixes(values: Tensor) -> dict:
    checked = {}
    for dtype in WireDtype:
        cpu = values.to(dtype.torch_dtype).cpu()
        for count in range(1, MAX_K + 1):
            raw = serialize_cpu(cpu[:, :count], dtype, 0)
            decoded = deserialize_continuous_tokens(raw)
            if decoded.end != count or decoded.values.shape[2] != values.shape[2]:
                raise ValueError("a legal integer prefix failed parsing")
        checked[dtype.value] = MAX_K
    return checked


@torch.inference_mode()
def run_wire_checks(inputs: Inputs, output: Path) -> None:
    plan = prefix_plan(inputs.config)
    receiver = ContinuousReceiver.from_model(inputs.model)
    decision = json_at(output / "ordering_decision.json")
    orders = [order for order in load_orders(output) if order.name.name in decision["wire_orders"]]
    rows, segments = [], []
    exhaustive = independent = None
    for video, frame, source in validation_frames(inputs, representative(inputs.cohort)):
        for precision in inputs.config["prefix"]["precisions"]:
            with compute(precision):
                full = encode_continuous(inputs.model, source)
                for order in orders:
                    values = order.apply(full)
                    for dtype in WireDtype:
                        row = _packet_audit(values, dtype, order, frame, plan)
                        rows.append(row | {"video": video, "frame": frame, "compute_dtype": precision})
                        if frame == 0:
                            segments.append(_incremental(values, receiver, order, dtype, frame, precision)
                                            | {"video": video, "frame": frame})
                    if precision == "bf16" and exhaustive is None and order.name.name == "O0":
                        exhaustive = _all_legal_prefixes(values)
                    if precision == "bf16" and independent is None and order.name.name == decision["primary_order"]:
                        independent = _independent(receiver, order, values, output)
        print(f"Wire contracts {video} frame {frame}", flush=True)
    if not exhaustive or not independent:
        raise ValueError("missing exhaustive or independent wire checks")
    write_json(output / "wire_prefix_audit.json", inputs.provenance() | {
        "status": "PASS", "payload_prefix_only": True, "header_bytes_per_packet": HEADER_BYTES,
        "tested_budgets": plan, "exhaustive_independently_parsed_counts": exhaustive, "cases": rows,
    })
    write_json(output / "wire_roundtrip_audit.json", inputs.provenance() | {
        "status": "PASS", "fp32_bit_exact": True, "fp16_bf16_equal_standard_cast": True,
        "wire_endianness": "little", "roundtrip_cases": len(rows), "incremental_cases": segments,
        "independent_process": independent,
    })
