"""Matched-history quantization, ordering, color and spatial-frequency diagnostics."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import functional as F

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.anchor import BUDGETS, AnchorModel, RefinementCodec, Segmentation
from hla_tok.uf.base import schedule
from hla_tok.uf.latent import LATENT_UNIT, UnifiedRefinement, normalized_error, render_latent
from hla_tok.uf.official import NativeDecoder, OfficialAssets
from hla_tok.uf_v2.records import BatchRecords, quantized_symbols
from hla_tok.uf_v2.runtime import Analysis
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v2.evaluate import QPS
from scripts.hla_tok.v2.protocol import setup
from third_party.DCVC.src.utils.transforms import rgb2ycbcr, ycbcr2rgb

RULE = Segmentation(8, 8)


def canvas_at(bank, base, symbols, cap):
    family = bank.pack_symbols(base, b"diagfam!", symbols, BUDGETS[40])
    raw, expected = family.prefix(cap)
    canvas = bank.canvas(tuple(symbols.shape[1:]), base.info.intra)
    added = raw[family.base_end:]
    if canvas.decode(bytearray(added))[:2] != (len(added), expected):
        raise ValueError("diagnostic prefix did not restore its integer symbols")
    return canvas, len(raw)


class OracleRecords(BatchRecords):
    """Offline first-order oracle with target RGB and LPIPS gradients."""

    score: np.ndarray

    def _priorities(self, statistics, counts, intra):
        normal, active, indexes = super()._priorities(statistics, counts, intra)
        relative_cost_inverse = normal / np.maximum(statistics[..., 1] / np.array([[1], [4]]), 1)
        return self.score * relative_cost_inverse, active, indexes


def oracle_score(lpips, decoder, base, reference, y, ctx, quant_step, symbols, shape):
    with torch.enable_grad():
        latent = y.float().requires_grad_(True)
        prediction = render_latent(decoder, base.info, latent.half().contiguous(memory_format=torch.channels_last), ctx)
        loss = (prediction - reference).square().mean() + 0.025 * lpips(
            prediction * 2 - 1, reference * 2 - 1).mean()
        gradient, = torch.autograd.grad(loss, latent)
    coarse_fine = torch.from_numpy(symbols).to(y.device).float() * quant_step[0].float()
    coarse_fine[1].mul_(0.5)
    utility = (-gradient * coarse_fine).cpu().numpy()
    bands, rows, cols = shape.grid
    padded = np.pad(utility, ((0, 0), (0, bands * RULE.channels - utility.shape[1]),
                              (0, rows * RULE.tile - utility.shape[2]), (0, cols * RULE.tile - utility.shape[3])))
    return padded.reshape(2, bands, RULE.channels, rows, RULE.tile, cols, RULE.tile).sum(
        axis=(2, 4, 6)).transpose(0, 2, 3, 1).reshape(2, -1)


def components(image, rgb):
    diff = rgb2ycbcr(image) - rgb2ycbcr(rgb)
    energy = diff.square().mean((0, 2, 3)).tolist()
    e = image - rgb
    low = F.avg_pool2d(e, 9, stride=1, padding=4, count_include_pad=False)
    return {"y_mse": energy[0], "cb_mse": energy[1], "cr_mse": energy[2],
            "low_mse": low.square().mean().item(), "high_mse": (e - low).square().mean().item()}


def substitutions(rgb, output):
    y, uv = rgb2ycbcr(rgb).split((1, 2), dim=1)
    out_y, out_uv = rgb2ycbcr(output).split((1, 2), dim=1)
    diff = output - rgb
    low_error = F.avg_pool2d(diff, 9, stride=1, padding=4, count_include_pad=False)
    return {
        "Q8_true_luma": ycbcr2rgb(torch.cat((y, out_uv), dim=1)).clamp(0, 1),
        "Q8_true_chroma": ycbcr2rgb(torch.cat((out_y, uv), dim=1)).clamp(0, 1),
        "Q9_true_high_frequency": (rgb + low_error).clamp(0, 1),
        "Q9_true_low_frequency": (output - low_error).clamp(0, 1),
    }


def save_image(path, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    pixels = rgb.squeeze(0).permute(1, 2, 0).mul(255).round().byte().contiguous().cpu().numpy()
    if not cv2.imwrite(str(path), cv2.cvtColor(pixels, cv2.COLOR_RGB2BGR)):
        raise ValueError(f"failed to save sample: {path}")
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "dimensions_hw": list(pixels.shape[:2])}


def load_anchor(path):
    model = AnchorModel().cuda().eval()
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["model"])
    codec = RefinementCodec(model, "bf16")
    return codec, BatchRecords(codec.records_codec.scales, codec.records_codec.segmentation)


def anchor_variants(anchors, native_base, rgb, base, outputs, sizes):
    for seed, (codec, bank) in enumerate(anchors, 1):
        with codec.compute(rgb):
            analysis = codec.model.analyze(rgb, native_base)
        symbols = quantized_symbols(analysis, codec.step)
        for scenario, cap in [("Q2_full", BUDGETS[40]), ("Q3_5", BUDGETS[5]), ("Q3_10", BUDGETS[10])]:
            canvas, byte_count = canvas_at(bank, base, symbols, cap)
            with codec.compute(rgb):
                rec = codec.model.display(native_base, canvas.delta(codec.step))
            name = f"{scenario}_anchor_seed{seed}"
            outputs[name], sizes[name] = rec, byte_count


def diagnose_frame(metric, decoder, base, rgb, base_rgb, native, model, banks, anchors):
    intra = base.info.intra
    original, _, _ = native.proxy(intra).ufv2_encode_snapshot()
    latent, step, *ctxs = decoder.last_net.proxy.uf_display_snapshot()
    ctx = ctxs[0] if ctxs else None
    continuous = original.float() * step.float()
    unit = model.quantizer(intra).step()
    symbols = quantized_symbols(normalized_error(original, latent, step), unit)
    quant = step.float() * unit * LATENT_UNIT
    def native_format(y):
        return y.half().contiguous(memory_format=torch.channels_last)

    outputs = {"Q1_base": base_rgb,
               "Q4_continuous_bypass": render_latent(decoder, base.info, native_format(continuous), ctx)}
    sizes = {"Q1_base": len(base.payload) + 30}
    quant_loss = {}
    ranked, fixed, oracle = banks
    oracle.score = oracle_score(metric.suite.networks["lpips"], decoder, base, rgb, latent, ctx, quant, symbols,
                                oracle.geometry(tuple(symbols.shape[1:])).groups)
    for scenario, bank, cap in [("Q5_d0_full", ranked, BUDGETS[40]), ("Q5_d0_5", ranked, BUDGETS[5]),
                                ("Q6_fixed_5", fixed, BUDGETS[5]), ("Q7_oracle_5", oracle, BUDGETS[5])]:
        integer, sizes[scenario] = canvas_at(bank, base, symbols, cap)
        display_y = latent.float() + integer.delta(quant)
        quant_loss[scenario] = (continuous - display_y).square().mean().item()
        outputs[scenario] = render_latent(decoder, base.info, native_format(display_y), ctx)
    anchor_variants(anchors, base_rgb, rgb, base, outputs, sizes)
    outputs.update(substitutions(rgb, outputs["Q3_5_anchor_seed1"]))
    metric.prepare(rgb)
    rows = [{"scenario": scenario, **metric.score(image), **components(image, rgb),
             "physical_bytes": sizes.get(scenario) if not scenario.startswith("Q7") else None,
             "latent_mse_to_continuous": quant_loss.get(scenario)} for scenario, image in outputs.items()]
    return rows, outputs


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, nargs=2, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup()
    reader, plans = inputs()
    assets = OfficialAssets.verify(args.weights)
    native, decoder = Analysis(assets), NativeDecoder(assets)
    decoder.intra.requires_grad_(False)
    decoder._video().requires_grad_(False)
    model, metric = UnifiedRefinement().cuda().eval(), NativeQuality()
    scales = tuple(model.scales().tolist())
    banks = BatchRecords(scales, RULE), BatchRecords(scales, Segmentation(8, 8, ranked=False)), OracleRecords(scales, RULE)
    anchors = [load_anchor(path) for path in args.anchor]
    rows, images = [], []
    indices = (0, 1, 5, 11)
    for clip in plans["validation"]:
        native.reset()
        decoder.reset()
        for i in range(indices[-1] + 1):
            rgb = source_frame(reader, clip, i)
            intra, reset = schedule(i)
            base = native.encode_base(rgb, i, QPS, intra, reset, 9212)
            base_rgb = decoder.decode(base)
            if i in indices:
                stats, outputs = diagnose_frame(metric, decoder, base, rgb, base_rgb, native, model, banks, anchors)
                rows.extend({"video": clip.video, "frame": i, "intra": intra, **row} for row in stats)
                if i == 5 and clip == plans["validation"][0]:
                    names = ("Q1_base", "Q3_5_anchor_seed1", "Q5_d0_5", "Q4_continuous_bypass", "Q9_true_high_frequency")
                    for name, image in [("source", rgb), *((k, outputs[k]) for k in names)]:
                        images.append(save_image(args.output / "samples" / f"{name}.png", image))
    write_csv(args.output / "frames.csv", rows)
    write_json(args.output / "samples/manifest.json", {"samples": images, "original_hw": [1080, 1920],
               "clip": plans["validation"][0].video, "frame": 5})
    scenarios = {name: summarize([r for r in rows if r["scenario"] == name]) for name in {r["scenario"] for r in rows}}
    write_json(args.output / "summary.json", {"scenarios": dict(sorted(scenarios.items())), "read_counts": reader.counts,
               "scope": "12 native Validation frames scored, 36 sequential frames decoded; Q0 points in official archive",
               "oracle": "first-order target-conditioned MSE + LPIPS gradient, offline diagnostic"})


if __name__ == "__main__":
    main()
