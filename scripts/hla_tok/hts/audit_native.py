"""Sweep actual official DCVC-UF streams on the frozen Calibration clips."""

import argparse
import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

import torch
from torch import Tensor

from hla_tok.evaluation.data import AuditedFrames
from hla_tok.evaluation.io import environment, write_csv, write_json
from hla_tok.evaluation.metrics import metric_hashes, sha256_file
from hla_tok.hts.native import NativeNetworks, restore_native_rgb, timed
from hla_tok.hts.units import NativeSchedule, Unit
from hla_tok.progressive.geometry import NATIVE_SHAPE
from hla_tok.uf.official import NativeNAL
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.temporal.inputs import Clip
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.v2.protocol import COHORT, json_at
from third_party.DCVC.src.utils.common import set_torch_env

VARIANTS = ("ld", "hts", "htl")
PROFILES = ("official", "matched")
CALIBRATION_FRAMES = 128
DEFAULT_OUTPUT = Path("results/hla_tok/hts/native_audit")
UNIDENTIFIABLE = "NOT_IDENTIFIABLE"


@dataclass(frozen=True)
class Point:
    profile: str
    qp_i: int
    qp_p: int

    @property
    def key(self) -> str:
        return f"{self.profile}-i{self.qp_i:02}-p{self.qp_p:02}"

    def qp(self, unit: Unit) -> int:
        return self.qp_i if unit.intra else self.qp_p


@dataclass(frozen=True)
class Source:
    clip: Clip
    fps: float
    rgb: Tensor


def source_clips(reader: AuditedFrames, clips: list[Clip]) -> list[Source]:
    result = []
    for clip in clips:
        spec = reader.catalog(clip.split)[clip.video]
        if spec["resized"] or tuple(reversed(spec["resolution"])) != NATIVE_SHAPE:
            raise ValueError(f"{clip.video} must be native 1080p")
        frames = torch.stack([reader.read(clip.split, clip.video, clip.start + i) for i in range(clip.length)])
        rgb = frames.cuda().float().div_(255).contiguous(memory_format=torch.channels_last)
        result.append(Source(clip, spec["fps"], rgb))
    return result


def profile_schedule(variant: str, profile: str) -> tuple[NativeSchedule, float]:
    delay = 1 if variant == "ld" else 8
    if profile == "official":
        return NativeSchedule(delay, reset_interval=128), 0.15
    return NativeSchedule(delay, gop_frames=32 if delay == 1 else 33, reset_interval=32), 0.0


def encode_clip(models: NativeNetworks, source: Source, units: list[Unit], point: Point):
    writer = NativeNAL()
    stream = bytearray()
    ledger = []
    for unit in units:
        qp = point.qp(unit)
        base, encoding_ms = timed(models.encode, source.rgb, unit, qp)
        packet = writer.encode(base)
        if not base.payload:
            raise ValueError("official encoder returned an empty payload")
        ledger.append({
            "sequence_id": source.clip.video,
            "source_fps": source.fps,
            "frame_ids": [source.clip.start + i for i in unit.indices],
            "unit_type": "I" if unit.intra else "P",
            "valid_frames": unit.valid,
            "padded_frames": unit.coded - unit.valid,
            "qp_requested": qp,
            "qp_effective": base.info.qp,
            "memory_reset": unit.reset,
            "native_payload_bytes": len(base.payload),
            "header_bytes": len(packet) - len(base.payload),
            "main_latent_bytes": UNIDENTIFIABLE,
            "hyperprior_bytes": UNIDENTIFIABLE,
            "offset": len(stream),
            "total_bytes": len(packet),
            "encoding_ms": encoding_ms,
        })
        stream.extend(packet)
    return bytes(stream), ledger


class CachedQuality(NativeQuality):
    """Reuse the upstream reference trunk outputs across QPs of one source clip."""

    def bind(self, rgb: Tensor) -> None:
        self.references: list[tuple[Tensor, list[Tensor]]] = []
        for image in rgb.split(1):
            self.prepare(image)
            self.references.append((self.original, self.dists.reference_features))

    def score_at(self, index: int, image: Tensor) -> dict:
        self.original, self.dists.reference_features = self.references[index]
        self.dists.reference = self.original
        return self.score(image)


def decode_clip(models: NativeNetworks, source: Source, units: list[Unit], stream: bytes,
                ledger: list[dict], quality: CachedQuality) -> list[dict]:
    reader = NativeNAL()
    height, width = source.rgb.shape[-2:]
    frame_metrics = []
    for unit, row in zip(units, ledger, strict=True):
        offset = row["offset"]
        base = reader.decode(stream[offset:offset + row["total_bytes"]], unit.start)
        yuv, decoding_ms = timed(models.decode, base)
        if len(yuv) != unit.coded:
            raise ValueError("official decoder returned an unexpected temporal shape")
        rgb = restore_native_rgb(yuv, unit, height, width)
        stats = [quality.score_at(index, image) for index, image in zip(unit.indices, rgb.split(1), strict=True)]
        frame_metrics.extend(stats)
        row.update(summarize(stats), decoding_ms=decoding_ms, reference_state_hash=models.reference_hash(unit))
    return frame_metrics


def aggregate(sequences: list[Source], rows: list[dict], frame_metrics: list[dict]) -> dict:
    summary = summarize(frame_metrics)
    frames, nbytes = summary["frames"], sum(row["total_bytes"] for row in rows)
    if frames != sum(s.clip.length for s in sequences) or frames != sum(row["valid_frames"] for row in rows):
        raise ValueError("valid frame accounting differs from the frozen source")
    elapsed_sec = sum(s.clip.length / s.fps for s in sequences)
    types = {kind: [row for row in rows if row["unit_type"] == kind] for kind in ("I", "P")}
    return summary | {
        "actual_mbps": 8 * nbytes * 30 / frames / 1e6,
        "actual_mbps_source_fps": 8 * nbytes / elapsed_sec / 1e6,
        "duration_source_fps_seconds": elapsed_sec,
        "bpp": 8 * nbytes / (frames * NATIVE_SHAPE[0] * NATIVE_SHAPE[1]),
        "total_bytes": nbytes,
        "native_payload_bytes": sum(row["native_payload_bytes"] for row in rows),
        "header_bytes": sum(row["header_bytes"] for row in rows),
        "padded_frames": sum(row["padded_frames"] for row in rows),
        "encode_fps": 1000 * frames / sum(row["encoding_ms"] for row in rows),
        "decode_fps": 1000 * frames / sum(row["decoding_ms"] for row in rows),
        "i_peak_bytes": max(row["total_bytes"] for row in types["I"]),
        "p_peak_bytes": max((row["total_bytes"] for row in types["P"]), default=0),
        "mean_unit_lpips": mean(row["lpips"] for row in rows),
    }


def atomic_bytes(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".tmp")
    pending.write_bytes(raw)
    pending.replace(path)


def run_clip(models: NativeNetworks, schedule: NativeSchedule, point: Point, source: Source,
             quality: CachedQuality, directory: Path, index: int) -> None:
    record = directory / f"{index:02}.json"
    bitstream = record.with_suffix(".bin")
    if record.is_file():
        prior = json_at(record)
        if prior["bitstream_sha256"] != sha256_file(bitstream):
            raise ValueError(f"changed native bitstream: {bitstream}")
        return
    units = list(schedule.units(source.clip.length))
    stream, ledger = encode_clip(models, source, units, point)
    metrics = decode_clip(models, source, units, stream, ledger, quality)
    atomic_bytes(bitstream, stream)
    write_json(record, {"bitstream_sha256": hashlib.sha256(stream).hexdigest(),
                        "metrics": metrics, "units": ledger})


def finish_point(variant: str, point: Point, sources: list[Source], root: Path) -> dict:
    directory = root / variant / point.key
    units: list[dict] = []
    metrics: list[dict] = []
    for index, source in enumerate(sources):
        path = directory / f"{index:02}.json"
        outcome = json_at(path)
        bitstream = path.with_suffix(".bin")
        source_units = outcome["units"]
        if sum(row["total_bytes"] for row in source_units) != bitstream.stat().st_size:
            raise ValueError(f"native byte ledger disagrees with {bitstream}")
        if [frame for row in source_units for frame in row["frame_ids"]] != list(
                range(source.clip.start, source.clip.start + source.clip.length)):
            raise ValueError("the native ledger omitted or reordered source frames")
        units.extend(row | {"model": variant, "profile": point.profile, "point": point.key,
                           "stream": str(bitstream.relative_to(root))} for row in source_units)
        metrics.extend(outcome["metrics"])
    summary = {"model": variant, "profile": point.profile, "qp_i": point.qp_i, "qp_p": point.qp_p,
               **aggregate(sources, units, metrics)}
    write_json(directory / "point.json", {"summary": summary, "units": units})
    return summary


def point_grid(profile: str, qps: list[int], mixed: bool) -> Iterable[Point]:
    yield from (Point(profile, qp, qp) for qp in qps)
    if mixed:
        grid = (0, 4, 8, 12)
        yield from (Point(profile, i, p) for i in grid for p in grid if i != p)


def run_profile(args, variant: str, profile: str, sources: list[Source], quality: CachedQuality) -> None:
    schedule, skip = profile_schedule(variant, profile)
    todo = [point for point in point_grid(profile, args.qps, args.mixed and variant == "hts")
            if not (args.output / variant / point.key / "point.json").is_file()]
    if not todo:
        return
    with torch.cuda.stream(torch.cuda.Stream()):
        models = NativeNetworks(args.weights, variant, skip)
        for index, source in enumerate(sources):
            quality.bind(source.rgb)
            for point in todo:
                run_clip(models, schedule, point, source, quality, args.output / variant / point.key, index)
                print(variant, point.key, source.clip.video, flush=True)
            quality.references.clear()
    torch.cuda.synchronize()
    for point in todo:
        report = finish_point(variant, point, sources, args.output)
        print(variant, point.key, f"{report['actual_mbps']:.5f} Mbps",
              f"{report['global_psnr']:.3f} dB LPIPS {report['lpips']:.5f}", flush=True)


def consolidate(output: Path) -> None:
    points = [json_at(path) for path in sorted(output.glob("*/*/point.json"))]
    write_csv(output / "qp_sweep.csv", [point["summary"] for point in points])
    atomic_bytes(output / "byte_ledger.jsonl", b"".join(
        json.dumps(row, allow_nan=False).encode() + b"\n" for point in points for row in point["units"]))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", choices=VARIANTS, nargs="+", default=VARIANTS)
    parser.add_argument("--profiles", choices=PROFILES, nargs="+", default=PROFILES)
    parser.add_argument("--qps", type=int, choices=range(64), nargs="+", default=list(range(64)))
    parser.add_argument("--mixed", action="store_true")
    return parser.parse_args()


@torch.inference_mode()
def main() -> None:
    args = parse_args()
    set_torch_env()
    reader, plans = inputs()
    sources = source_clips(reader, plans["calibration"])
    if sum(s.clip.length for s in sources) != CALIBRATION_FRAMES or reader.counts["validation"]:
        raise ValueError("Phase 0 requires exactly the frozen Calibration clips")
    expected = json_at(COHORT)["metric_weights"]
    if metric_hashes() != expected:
        raise ValueError("perceptual weights differ from historical experiments")
    quality = CachedQuality()
    write_json(args.output / "pixel_reads.json", reader.counts)
    write_json(args.output / "runtime.json", environment())
    for variant in args.models:
        for profile in args.profiles:
            run_profile(args, variant, profile, sources, quality)
    consolidate(args.output)


if __name__ == "__main__":
    main()
