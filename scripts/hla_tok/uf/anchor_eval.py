"""Real bitstream Anchor evaluation with one family per native frame."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from time import perf_counter
from typing import Protocol

import torch
from torch import Tensor

from hla_tok.evaluation.data import AuditedFrames
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.anchor import (
    BUDGETS,
    MAX_APPLICATION_BYTES,
    AnchorModel,
    FrameReceiver,
    ProgressiveReceiver,
    RefinementCodec,
)
from hla_tok.uf.base import schedule
from hla_tok.uf.official import NativeDecoder, NativeEncoder, OfficialAssets
from hla_tok.uf.wire import APP_BASE_OVERHEAD, RECORD_OVERHEAD, BaseBudgetExceeded, PrefixFamily
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.temporal.inputs import Clip, source_frame
from scripts.hla_tok.uf.baseline import BASE_CAP, RATE_FPS, distribution
from scripts.hla_tok.uf.inputs import OUTPUT, checkpoint_directory, inputs
from scripts.hla_tok.v2.protocol import json_at, setup


@dataclass(frozen=True)
class Encoded:
    family: PrefixFamily
    reference: Tensor
    phase_ms: dict[str, float]


class FrameEncoder(Protocol):
    def reset(self) -> None: ...

    def encode(self, rgb: Tensor, index: int, intra: bool, reset: bool = False) -> Encoded: ...

    def receiver(self, decoder: NativeDecoder) -> FrameReceiver: ...

    def matches(self, encoded: Encoded, decoder: NativeDecoder, receiver: FrameReceiver) -> bool: ...


class AnchorEncoder:
    def __init__(self, assets: OfficialAssets, codec: RefinementCodec, qp: tuple[int, int]):
        self.encoder = NativeEncoder(assets)
        self.preview = NativeDecoder(assets)
        self.qp = qp
        self.codec = codec

    def reset(self) -> None:
        self.encoder.reset()
        self.preview.reset()

    def receiver(self, decoder: NativeDecoder) -> FrameReceiver:
        return ProgressiveReceiver(self.codec, decoder.decode)

    def matches(self, encoded: Encoded, decoder: NativeDecoder, receiver: FrameReceiver) -> bool:
        return receiver.base is not None and torch.equal(encoded.reference, receiver.base)

    @torch.no_grad()
    def encode(self, rgb: Tensor, index: int, intra: bool, reset: bool = False) -> Encoded:
        start = perf_counter()
        base = self.encoder.encode(rgb, index, self.qp[0 if intra else 1], intra, reset)
        encoded = perf_counter()
        preview = self.preview.decode(base)
        previewed = perf_counter()
        with self.codec.compute(rgb):
            latent = self.codec.model.analyze(rgb, preview)
        torch.cuda.current_stream().synchronize()
        analyzed = perf_counter()
        family = PrefixFamily.build(base, self.codec.tag, self.codec.records(latent), MAX_APPLICATION_BYTES)
        finalized = perf_counter()
        return Encoded(family, preview, {"native_encode_ms": (encoded - start) * 1e3,
                       "analysis_base_decode_ms": (previewed - encoded) * 1e3,
                       "analysis_ms": (analyzed - previewed) * 1e3,
                       "quant_rank_entropy_pack_ms": (finalized - analyzed) * 1e3,
                       "encode_ms": (finalized - start) * 1e3})


def endpoint_rows(enc: FrameEncoder, decoders: dict[int, NativeDecoder], quality: NativeQuality | None,
                  rgb: Tensor, clip: Clip, index: int) -> list[dict]:
    intra, reset = schedule(index)
    if quality is not None:
        quality.prepare(rgb)
    torch.cuda.synchronize()
    encoded = enc.encode(rgb, index, intra, reset)
    family = encoded.family
    rows = []
    for bandwidth, decoder in decoders.items():
        cap = BUDGETS.get(bandwidth, family.base_end)
        row = {"video": clip.video, "source_index": clip.start + index, "frame": index,
               "type": "I" if intra else "P", "budget_mbps": bandwidth, "cap": cap,
               "base_bytes": family.base_end, "base_codec_bytes": family.base_end - APP_BASE_OVERHEAD,
               "base_header_bytes": APP_BASE_OVERHEAD, **encoded.phase_ms}
        rows.append(row | _decode_budget(enc, decoder, encoded, cap, quality))
    return rows


def _decode_budget(encoder: FrameEncoder, decoder: NativeDecoder, encoded: Encoded, cap: int,
                   quality: NativeQuality | None) -> dict:
    start = perf_counter()
    family = encoded.family
    try:
        raw, token_count = family.prefix(cap)
    except BaseBudgetExceeded:
        decoder.order.desync = True
        return {"status": "BASE_BUDGET_EXCEEDED"}
    try:
        state = encoder.receiver(decoder)
        state.feed(raw)
        state.decode_available()
        state.finish()
    except ValueError as error:
        if "DESYNC" not in str(error):
            raise
        return {"status": "DESYNC", "bytes": len(raw)}
    restored = perf_counter()
    display = state.render()
    torch.cuda.current_stream().synchronize()
    end = perf_counter()
    exact_base = encoder.matches(encoded, decoder, state)
    scores = quality.score(display) if quality is not None else {}
    return scores | {"status": "PASS" if exact_base else "StateMismatch", "bytes": len(raw),
                     "unused_bytes": cap - len(raw), "tokens": token_count, "symbols": state.symbol_count,
                     "enhancement_headers": token_count * RECORD_OVERHEAD,
                     "enhancement_codec_bytes": len(raw) - family.base_end - token_count * RECORD_OVERHEAD,
                     "parse_decode_update_ms": 1e3 * (restored - start), "display_ms": 1e3 * (end - restored),
                     "decode_ms": 1e3 * (end - start), "total_ms": encoded.phase_ms["encode_ms"] + 1e3 * (end - start)}


def anchor_summary(rows: list[dict]) -> list[dict]:
    result = []
    for budget in sorted({row["budget_mbps"] for row in rows}):
        group = [row for row in rows if row["budget_mbps"] == budget]
        ok = [row for row in group if row["status"] == "PASS"]
        if len(ok) != len(group):
            result.append({"budget_mbps": budget, "status": "FAIL", "frames_scored": len(ok),
                           "frames_planned": len(group), "base_failures": sum(
                               row["status"] == "BASE_BUDGET_EXCEEDED" for row in group)})
            continue
        result.append(_complete_summary(group) | {"budget_mbps": budget})
    return result


def _complete_summary(group: list[dict]) -> dict:
    stats = summarize(group) if "mse" in group[0] else {"frames": len(group)}
    stats |= {"status": "PASS", "actual_mbps": mean(row["bytes"] for row in group) * 8 * RATE_FPS / 1e6,
              "actual_bpp": mean(row["bytes"] for row in group) * 8 / (1920 * 1080),
              "serial_fps": 1000 / mean(row["total_ms"] for row in group),
              "base_max": max(row["base_bytes"] for row in group),
              "base_oversized_frames": sum(row["base_bytes"] > BASE_CAP for row in group)}
    for key in ("bytes", "unused_bytes", "tokens", "symbols", "base_bytes", "total_ms", "encode_ms",
                "decode_ms", "native_encode_ms", "analysis_base_decode_ms", "analysis_ms",
                "quant_rank_entropy_pack_ms", "parse_decode_update_ms", "display_ms"):
        stats |= distribution((row[key] for row in group), key)
    return stats


def evaluate_encoder(assets: OfficialAssets, encoder: FrameEncoder, reader: AuditedFrames,
                     clips: list[Clip], output: Path, quality: NativeQuality | None, metadata: dict) -> dict:
    decoders = {budget: NativeDecoder(assets) for budget in (0, *BUDGETS)}
    for index in range(4):
        endpoint_rows(encoder, decoders, None, source_frame(reader, clips[0], index), clips[0], index)
    rows = []
    for clip in clips:
        encoder.reset()
        for decoder in decoders.values():
            decoder.reset()
        for index in range(clip.length):
            rows.extend(endpoint_rows(encoder, decoders, quality, source_frame(reader, clip, index), clip, index))
        print(metadata["precision"], clip.video, clip.length, flush=True)
    write_csv(output / "per_frame.csv", rows)
    report = metadata | {"rows": anchor_summary(rows), "warmup_frames": 4,
              "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "read_counts": reader.counts,
              "base_by_type": {kind: distribution((row["base_bytes"] for row in rows
                                                     if row["type"] == kind and not row["budget_mbps"]), "bytes")
                               for kind in ("I", "P")}}
    write_json(output / "summary.json", report)
    return report


def evaluate(assets: OfficialAssets, model: AnchorModel, precision: str, qp: tuple[int, int],
             reader: AuditedFrames, clips: list[Clip], output: Path, quality: NativeQuality | None) -> dict:
    torch.cuda.reset_peak_memory_stats()
    codec = RefinementCodec(model.eval(), precision)
    metadata = {"precision": f"FP16 native + {precision.upper()} enhancement", "base_qp": qp,
                "model_tag": codec.tag.hex(),
                "enhancement_parameters": sum(parameter.numel() for parameter in model.parameters())}
    return evaluate_encoder(assets, AnchorEncoder(assets, codec, qp), reader, clips, output, quality, metadata)


def locked_qp(path: Path) -> tuple[int, int]:
    score = json_at(path)["score"]
    return score["qp_i"], score["qp_p"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT / "anchor/A0-bf16")
    parser.add_argument("--base-lock", type=Path, default=OUTPUT / "baseline/base_lock.json")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--split", choices=("calibration", "validation", "long"), default="validation")
    args = parser.parse_args()
    setup()
    reader, plans = inputs()
    model = AnchorModel().cuda()
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True)["model"], strict=True)
    evaluate(OfficialAssets.verify(checkpoint_directory(args.weights)), model, args.precision,
             locked_qp(args.base_lock), reader, plans[args.split], args.output, NativeQuality())


if __name__ == "__main__":
    main()
