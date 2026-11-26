"""H100 study on audited Calibration RGB and an external UVG segment."""

import argparse
import csv
import json
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from cp_adaptok.metrics import MetricSuite
from hla_tok.evaluation.metrics import metric_hashes
from hla_tok.hts.native import NativeNetworks, frame_input, timed
from hla_tok.hts.units import NativeSchedule, Unit
from hla_tok.uf.official import NativeNAL
from pht_tok.bitstream import Mode
from pht_tok.codec import PHTDecoder, PHTEncoder, ReferenceKey, model_identifier
from pht_tok.official_adapter import EncodedLatents, HTSAdapter, Reference
from scripts.hla_tok.hts.audit_native import source_clips
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.v2.protocol import COHORT
from third_party.DCVC.src.utils.common import set_torch_env
from third_party.DCVC.src.utils.transforms import rgb2ycbcr, ycbcr2rgb, ycbcr420_to_444_np
from third_party.DCVC.src.utils.video_reader import YUV420Reader

ROOT = Path(__file__).resolve().parents[2]
QPS = (0, 16, 32, 48, 63)
HEIGHT, WIDTH = 1080, 1920
SKIP = 0.15
MODES = (Mode.MASK, Mode.PRECISION)


@dataclass(frozen=True, slots=True)
class Source:
    name: str
    fps: float
    rgb: Tensor
    yuv: Tensor


def calibration() -> tuple[list[Source], dict]:
    reader, plans = inputs()
    result = [Source(item.clip.video, item.fps, item.rgb, rgb2ycbcr(item.rgb).half() - 0.5)
              for item in source_clips(reader, plans["calibration"])]
    return result, reader.counts


def uvg_source(path: Path, count: int) -> Source:
    if path.name != "ShakeNDry_1920x1080_120fps_420_8bit_YUV.yuv":
        raise ValueError("expected the published UVG ShakeNDry asset")
    with YUV420Reader(str(path), WIDTH, HEIGHT) as reader:
        frames = [ycbcr420_to_444_np(*reader.read_one_frame()) for _ in range(count)]
    yuv = torch.from_numpy(np.stack(frames)).cuda().half() / 255 - 0.5
    rgb = ycbcr2rgb(yuv.float() + 0.5)
    return Source("UVG:ShakeNDry", 120, rgb, yuv)


def write_json(path: Path, data) -> None:
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(data, allow_nan=False, indent=2) + "\n")
    pending.replace(path)


def exact(left, right, field):
    if not torch.equal(left, right):
        raise ValueError(f"native equivalence failed for {field}")


def verify_conditions(adapter: HTSAdapter, expected: list[Tensor], qp: int):
    for key, a, b in zip(("q_encoder", "q_decoder", "q_feature", "prior", "means_3", "y_hat", "z_hat"),
                         expected, adapter.proxy.pht_conditions(qp), strict=True):
        exact(a, b, key)


def training_math_probe(adapter: HTSAdapter, frames: Tensor, qp: int) -> dict:
    """Compare the readable training operations against the native analysis."""
    proxy, model = adapter.proxy, adapter.model
    adapter._prepare(qp)
    bottom, right = (-frames.shape[-2]) % 16, (-frames.shape[-1]) % 16
    native_y, native_z, *spatial_means = proxy.pht_audit_analysis(frames, qp, bottom, right)
    hyper, symbols, _, _ = proxy.pht_symbols(True)
    qe, _, qf, prior, *_ = proxy.pht_conditions(qp)
    memory, ctx = proxy.audit_context()
    py_y = model.encoder(F.pad(frames, (0, right, 0, bottom), mode="replicate"), ctx, qe)
    py_z = model.hyper_encoder(F.pad(py_y, (0, (-py_y.shape[3]) % 4, 0, (-py_y.shape[2]) % 4),
                                    mode="replicate"))
    condition = model.res_prior_param_decoder(py_z.round().clamp(-128, 127), memory, qf)
    _, py_symbols, _, _ = model.forward_prior_4x(
        py_y, None, None, condition, model.y_spatial_prior_reduction, model.y_spatial_prior_adaptor_1,
        model.y_spatial_prior_adaptor_2, model.y_spatial_prior_adaptor_3, model.y_spatial_prior,
        spatial_prior_has_scales=False
    )
    return {"training_reference": "native prepared memory and context",
            "native_y_shape": list(native_y.shape), "native_z_shape": list(native_z.shape),
            "maximum_unquantized_y_abs_difference": float((native_y - py_y).abs().max()),
            "maximum_unquantized_z_abs_difference": float((native_z - py_z).abs().max()),
            "z_integer_differences": int(hyper.ne(py_z.round().clamp(-128, 127)).sum()),
            "y_integer_differences": int(symbols.ne(py_symbols).sum()),
            "spatial_means_shapes": [list(item.shape) for item in spatial_means],
            "native_prior_shape": list(prior.shape)}


def equivalence(sources: list[Source], qps: list[int], weights: Path, work: Path) -> dict:
    stock = NativeNetworks(weights, "hts", SKIP)
    tx = HTSAdapter(NativeNetworks(weights, "hts", SKIP).video)
    rx = HTSAdapter(NativeNetworks(weights, "hts", SKIP).video)
    rows = []
    probe = None
    for source in sources:
        units = list(NativeSchedule(8, reset_interval=128).units(len(source.rgb)))
        for qp in qps:
            nal = NativeNAL()
            official = [stock.encode(source.rgb, unit, qp) for unit in units]
            stream = b"".join(nal.encode(packet) for packet in official)
            archived = work / "Calibration-hts/streams/Calibration" / f"{source.name.split(':')[-1]}_q{qp}.bin"
            if archived.is_file() and stream != archived.read_bytes():
                raise ValueError("PHT extension changed the original official full stream")
            first = stock.decode(official[0])[0].clone()
            tx.initialize(first)
            rx.initialize(first)
            for unit, packet in zip(units[1:], official[1:], strict=True):
                x = frame_input(source.rgb, unit)
                analysis = tx.encode_chunk(x, qp)
                conditions = [item.clone() for item in tx.proxy.pht_conditions(qp)]
                indexes, skip = rx.start_decoding(analysis.hyper, qp)
                exact((analysis.packed & 255).to(torch.uint8), indexes, "Y CDF indexes")
                exact(analysis.keep, skip, "skip mask")
                for group in range(4):
                    rx.apply_group(analysis.residuals, group)
                verify_conditions(rx, conditions, qp)
                actual = rx.reconstruct(4, qp, reset=unit.reset)
                expected = stock.decode(packet)
                z, y, *_ = stock.video.proxy.pht_symbols(False)
                exact(analysis.hyper, z, "encoded vs independently decoded Z integers")
                exact(analysis.residuals, y, "encoded vs independently decoded Y integers")
                for a, b in zip(actual, expected, strict=True):
                    exact(a, b, "eight reconstructed frames")
                for a, b in zip(rx.proxy.audit_context(), stock.video.proxy.audit_context(), strict=True):
                    exact(a, b, "prepared reference memory/context")
                if probe is None:
                    probe = training_math_probe(tx, x, qp)
                    analysis = tx.encode_chunk(x, qp)
                for group in range(4):
                    tx.apply_group(analysis.residuals, group)
                tx.reconstruct(4, qp, reset=unit.reset, display=False)
                rx_state, tx_state = rx.snapshot_reference(unit.reset), tx.snapshot_reference(unit.reset)
                for a, b, c in zip(rx_state.tensors, tx_state.tensors,
                                   stock.video.proxy.audit_reference(unit.reset), strict=True):
                    exact(a, b, "independent PHT reference")
                    exact(b, c, "official reference")
                tx.commit_reference(tx_state)
                rx.commit_reference(rx_state)
                rows.append({"source": source.name, "qp": qp, "start": unit.start,
                             "valid_frames": unit.valid, "reset": unit.reset, "native_payload": len(packet.payload),
                             "native_stream_sha256": sha256(stream).hexdigest(), "ref": rx_state.digest(),
                             "z_shape": list(z.shape), "y_shape": list(y.shape), "exact": True})
            print("equivalence", source.name, qp, flush=True)
    return {"status": "PASS", "fp16_tolerances": [0, 0], "confirmed_chunks": len(rows),
            "per_chunk": rows, "python_training_comparison": probe}


class Quality:
    def __init__(self, perceptual: bool = True):
        self.perceptual = perceptual
        if perceptual:
            if metric_hashes() != json.loads(COHORT.read_text())["metric_weights"]:
                raise ValueError("perceptual weights differ from the frozen source identity")
            self.metrics = MetricSuite("cuda", ("psnr", "lpips", "dists", "ms-ssim"))
            if self.metrics.errors:
                raise RuntimeError(self.metrics.errors)
            for name in ("lpips", "dists"):
                self.metrics.networks[name].to(memory_format=torch.channels_last)

    def score(self, source: Tensor, decoded: list[Tensor]) -> list[dict]:
        output = ycbcr2rgb(torch.cat(decoded)[..., :HEIGHT, :WIDTH] + 0.5).float()
        original = source.float().contiguous(memory_format=torch.channels_last)
        error = (output - original).square().mean(dim=(1, 2, 3))
        columns = {"mse": error, "psnr": -10 * error.clamp_min(1e-12).log10()}
        if self.perceptual:
            measured: dict[str, list[Tensor]] = {key: [] for key in ("lpips", "dists", "ms-ssim")}
            for a, b in zip(original.split(4), output.split(4), strict=True):
                for name in measured:
                    measured[name].append(self._score_metric(name, a, b).reshape(-1))
            columns.update({name: torch.cat(values) for name, values in measured.items()})
        return [dict(zip(columns, row, strict=True)) for row in torch.stack(tuple(columns.values())).T.tolist()]

    def _score_metric(self, name: str, a: Tensor, b: Tensor) -> Tensor:
        metric = self.metrics.networks[name]
        if name == "lpips":
            return metric(a * 2 - 1, b * 2 - 1)
        if name == "dists":
            return metric(a, b, batch_average=False)
        return metric(a, b, data_range=1.0, size_average=False)


def coding_frame(source: Source, unit: Unit) -> Tensor:
    return source.yuv[unit.start:unit.start + unit.valid]


def summarize(rows: list[dict]) -> dict:
    per_frame = [item for row in rows for item in row["frame_metrics"]]
    count = len(per_frame)
    bits = sum(row["bytes"] for row in rows) * 8
    seconds = sum(row["unit_valid"] / row["fps"] for row in rows)
    stats = {metric: mean(item[metric] for item in per_frame) for metric in per_frame[0]}
    return {"frames": count, "bytes": bits // 8, "bpp": bits / (count * HEIGHT * WIDTH),
            "mbps": bits / seconds / 1e6, "global_psnr": float(-10 * np.log10(stats["mse"])), **stats}


def ledger_row(source: Source, unit: Unit, qp: int, method: str, stage: int, size: int,
               frames: list[dict]) -> dict:
    return {"source": source.name, "fps": source.fps, "start": unit.start, "unit_valid": unit.valid,
            "tail_padding": unit.coded - unit.valid, "qp": qp, "intra": unit.intra, "reset": unit.reset,
            "method": method, "stage": stage, "bytes": size, "frame_metrics": frames}


def fixed_point(source, qp, stock, tx, rx, quality, sample_dir, estimate_rate=False) -> list[dict]:
    units = list(NativeSchedule(8, reset_interval=128).units(len(source.rgb)))
    from experiments.pht_tok.closed import official_encode
    writer, reader = NativeNAL(), NativeNAL()
    streams = [official_encode(stock, source, unit, qp, writer) for unit in units]
    control = [reader.decode(raw, unit.start) for raw, unit in zip(streams, units, strict=True)]
    physical = list(map(len, streams))
    tx.key = rx.key = None
    anchor = tx.encode_intra(source.yuv[:1], qp, 0)
    rx.decode_prefix(anchor.data)
    anchor_recon = stock.decode(control[0])
    anchor_score = quality.score(source.rgb[:1], anchor_recon)
    rows = [ledger_row(source, units[0], qp, "official", 4, physical[0], anchor_score)]
    for mode in MODES:
        rows.extend(ledger_row(source, units[0], qp, mode.name, stage, len(anchor.data), anchor_score)
                    for stage in range(1, 5))
    for index, (unit, packet, official_size) in enumerate(zip(units, control, physical, strict=True)):
        if not index:
            continue
        state = tuple(stock.video.proxy.audit_reference(units[index - 1].intra or units[index - 1].reset))
        ref = Reference(units[index - 1].intra or units[index - 1].reset, state)
        for peer in (tx, rx):
            peer.adapter.commit_reference(ref)
            peer.key = ReferenceKey(index - 1, 0, HEIGHT, WIDTH)
        expected = [frame.clone() for frame in stock.decode(packet)]
        exact_symbols = stock.video.proxy.pht_symbols(False)[1].clone()
        original = source.rgb[unit.start:unit.start + unit.valid]
        full_quality = quality.score(original, expected[:unit.valid])
        rows.append(ledger_row(source, unit, qp, "official", 4, official_size, full_quality))
        for mode in MODES:
            stream, enc_ms = timed(partial(tx.encode_chunk_once, mode=mode, reset=unit.reset, commit=False),
                                   coding_frame(source, unit), qp, index, 2)
            estimates = None
            if estimate_rate:
                from experiments.pht_tok.entropy_rate import estimate
                captured = EncodedLatents(*tx.adapter.proxy.pht_symbols(True))
                estimates = estimate(tx.entropy, captured, tx.layout, qp, mode)
            session = rx.begin_chunk(commit=False)
            previous = 0
            for stage, end in enumerate(stream.prefix_boundaries, 1):
                session.append_bytes(stream.data[previous:end])
                _, dec_ms = timed(session.decode_new_layers)
                output, render_ms = timed(session.reconstruct)
                scores = full_quality if stage == 4 else quality.score(original, output)
                if stage == 4:
                    exact(session.residuals(), exact_symbols, "physically decoded Full symbols")
                    for a, b in zip(output, expected, strict=False):
                        exact(a, b[..., :HEIGHT, :WIDTH], "physically decoded Full reconstruction")
                row = ledger_row(source, unit, qp, mode.name, stage, end, scores)
                row.update(enc_ms=enc_ms, incremental_entropy_ms=dec_ms, synthesis_ms=render_ms)
                if estimates is not None:
                    row["estimated_cumulative_payload_bits"] = sum(estimates[:stage + 1])
                if stage == 4:
                    row["ledger"] = stream.ledger
                    row["official_unit_bytes"] = official_size
                rows.append(row)
                previous = end
                if index == 1 and qp == 32 and sample_dir is not None:
                    sample_dir.mkdir(parents=True, exist_ok=True)
                    from PIL import Image
                    image = ycbcr2rgb(output[0] + 0.5)[0].permute(1, 2, 0).mul(255).round().byte().cpu().numpy()
                    Image.fromarray(image).save(sample_dir / f"{mode.name.lower()}-{stage}.jpg", quality=94)
            if index == 1 and qp == 32 and sample_dir is not None:
                (sample_dir / f"{mode.name.lower()}.pht").write_bytes(stream.data)
    return rows


def fixed(sources, qps, stock, tx, rx, quality, work):
    rows = []
    for i, source in enumerate(sources):
        for qp in qps:
            name = work / f"fixed-{i}-q{qp:02}.json"
            if name.is_file():
                results = json.loads(name.read_text())
            else:
                results = fixed_point(source, qp, stock, tx, rx, quality, work / "samples" if i == 0 else None)
                write_json(name, results)
            rows.extend(results)
            print("fixed", source.name, qp, flush=True)
    return rows


def aggregate_curves(rows):
    keys = sorted({(row["method"], row["qp"], row["stage"]) for row in rows})
    return [{"method": method, "qp": qp, "stage": stage, **summarize([
        row for row in rows if (row["method"], row["qp"], row["stage"]) == (method, qp, stage)
    ])} for method, qp, stage in keys]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("study", choices=["equivalence", "fixed", "closed"])
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=ROOT / ".build/pht-tok")
    parser.add_argument("--qps", type=int, choices=range(64), nargs="+", default=QPS)
    parser.add_argument("--uvg", type=Path)
    parser.add_argument("--sources", type=int, default=4)
    parser.add_argument("--psnr-only", action="store_true")
    args = parser.parse_args()
    set_torch_env()
    with torch.cuda.stream(torch.cuda.Stream()):
        if args.study == "equivalence":
            sources, counts = calibration()
            report = equivalence(sources[:args.sources], args.qps, args.weights, args.work)
            write_json(args.work / "latent_equivalence.json", report | {"pixel_reads": counts})
            return
        model_id = model_identifier(args.weights, SKIP)
        stock = NativeNetworks(args.weights, "hts", SKIP)
        tx = PHTEncoder(NativeNetworks(args.weights, "hts", SKIP), model_id)
        rx = PHTDecoder(NativeNetworks(args.weights, "hts", SKIP), model_id)
        quality = Quality(not args.psnr_only)
        if args.study == "fixed":
            sources, counts = calibration()
            rows = fixed(sources[:args.sources], args.qps, stock, tx, rx, quality, args.work)
            curves = aggregate_curves(rows)
            write_json(args.work / "fixed.json", {"curves": curves, "units": rows, "pixel_reads": counts})
            with (args.work / "rate_distortion.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, list(curves[0]), lineterminator="\n")
                writer.writeheader()
                writer.writerows(curves)
            return
        if args.uvg is None:
            raise ValueError("closed requires the official UVG 1080p development asset")
        from experiments.pht_tok.closed import closed_study
        source = uvg_source(args.uvg, 129)
        closed_study(source, args.qps, stock, tx, rx, quality, args.work)


if __name__ == "__main__":
    main()
