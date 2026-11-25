"""Fit static I/P enhancement CDFs and diagnostic perceptual sensitivity on Train holdout."""
from __future__ import annotations

import argparse
import hashlib
from collections.abc import Iterator
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from cp_adaptok.metrics import MetricSuite
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.anchor import BUDGETS, Segmentation
from hla_tok.uf.base import schedule
from hla_tok.uf.latent import UnifiedRefinement, normalized_error, render_latent
from hla_tok.uf.official import NativeDecoder, OfficialAssets
from hla_tok.uf_v2.records import BatchRecords, quantized_symbols
from hla_tok.uf_v2.runtime import Analysis
from hla_tok.uf_v2.static_entropy import Histogram
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v2.evaluate import QPS
from scripts.hla_tok.v2.protocol import setup

RULE = Segmentation(8, 8)


def samples(native, model, reader, clips) -> Iterator:
    for clip in clips:
        if clip.split != "train":
            raise PermissionError("CDF fitting accepts only the Train holdout")
        native.reset()
        for index in range(clip.length):
            rgb = source_frame(reader, clip, index)
            intra, reset = schedule(index)
            base = native.encode_base(rgb, index, QPS, intra, reset, 9212)
            original, latent, step = native.proxy(intra).ufv2_encode_snapshot()
            residual = normalized_error(original, latent, step)
            symbols = quantized_symbols(residual, model.quantizer(intra).step())
            yield clip.video, index, rgb, base, symbols


def sensitivity_sample(lpips, decoder, base, rgb):
    with torch.no_grad():
        assert decoder.last_net is not None
        latent, _, *cached = decoder.last_net.proxy.uf_display_snapshot()
        # 256×320 source crop and 128×192 inner loss. No spatial resizing.
        y, x = 20, 35
        latents = latent[..., y:y + 16, x:x + 20].float()
        context = None if not cached else cached[0][..., 2 * y:2 * y + 32, 2 * x:2 * x + 40].float()
        start_y, start_x = y * 16 + 64, x * 16 + 64
        target = rgb[..., start_y:start_y + 128, start_x:start_x + 192]
    with torch.enable_grad():
        latents = latents.requires_grad_()
        # Use the same FP16 weights and crop the decoded readout to the valid interior.
        ctx = context.half().contiguous(memory_format=torch.channels_last) if context is not None else None
        image = render_latent(decoder, base.info, latents.half().contiguous(memory_format=torch.channels_last), ctx)
        inner = image[..., 64:192, 64:256]
        mse = (inner - target).square().mean()
        lp = lpips(inner * 2 - 1, target * 2 - 1).mean()
        mse_grad, = torch.autograd.grad(mse, latents, retain_graph=True)
        lp_grad, = torch.autograd.grad(lp, latents)
        power = torch.stack([grad.square().mean((0, 2, 3)) for grad in (mse_grad, lp_grad)])
    return power.reshape(2, -1, RULE.channels).sum(-1).double().cpu().numpy()


def normalize(weights):
    power = np.stack(weights).mean(axis=0)
    return np.sqrt(power / np.maximum(power.mean(axis=-1, keepdims=True), 1e-30)).mean(axis=0).clip(0.25, 4)


def compare_sample(model, candidates, video, index, base, symbols):
    tag = b"entropy!"
    # E0 determines the reference support. E1/E2 are compared at the identical integer boundary.
    e0, *others = candidates
    families = []
    timings = []
    for i, bank in enumerate(candidates):
        tick = perf_counter()
        families.append(bank.pack_symbols(base, tag, symbols, BUDGETS[40] if i == 0 else (2 << 20)))
        timings.append((perf_counter() - tick) * 1000)
    rows = []
    for budget, cap in BUDGETS.items():
        _, stop = families[0].prefix(cap)
        totals = []
        reference = None
        for bank, family in zip(candidates, families, strict=True):
            raw = family.raw[family.base_end:family.ends[stop]]
            canvas = bank.canvas(tuple(symbols.shape[1:]), base.info.intra)
            consumed, received = canvas.decode(bytearray(raw))
            if (received, consumed) != (stop, len(raw)):
                raise ValueError("comparison did not decode the requested symbols")
            if reference is not None and not np.array_equal(reference, canvas.symbols):
                raise ValueError("CDF changed the integer field on fixed support")
            reference = canvas.symbols
            totals.append(len(raw) + family.base_end)
        rows.append({"video": video, "frame": index, "intra": base.info.intra, "budget": budget,
                     "base_bytes": families[0].base_end, "records": stop,
                     "e0_bytes": totals[0], "e1_bytes": totals[1], "e2_bytes": totals[2],
                     "one_max_e0_pack_ms": timings[0], "all_records_e1_pack_ms": timings[1],
                     "all_records_e2_pack_ms": timings[2]})
    return rows


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup()
    reader, plans = inputs()
    assets = OfficialAssets.verify(args.weights)
    native, decoder = Analysis(assets), NativeDecoder(assets)
    decoder.intra.requires_grad_(False)
    decoder._video().requires_grad_(False)
    lpips = MetricSuite("cuda", ("lpips",)).networks["lpips"].to(memory_format=torch.channels_last)
    model = UnifiedRefinement().cuda().eval()
    scales = tuple(model.scales().tolist())
    histogram = Histogram(RULE.channels)
    weights: dict[bool, list] = {False: [], True: []}
    for video, index, rgb, base, symbols in samples(native, model, reader, plans["calibration"]):
        if index == 0:
            decoder.reset()
        histogram.observe(base.info.intra, symbols)
        decoder.decode(base)
        if index % 8 == 0:
            weights[base.info.intra].append(sensitivity_sample(lpips, decoder, base, rgb))
    fitted = histogram.fit()
    write_json(args.output / "static_cdf.json", fitted.as_json())
    sensitivity = {key: normalize(values).tolist() for key, values in weights.items()}
    write_json(args.output / "sensitivity.json", {
        "I": sensitivity[True], "P": sensitivity[False],
        "definition": "half normalized RGB MSE gradient RMS plus half LPIPS-Alex gradient RMS by latent band",
        "samples": {"I": len(weights[True]), "P": len(weights[False])},
        "calibration_clips": [clip.video for clip in plans["calibration"]], "test_reads": 0})
    candidates = [BatchRecords(scales, RULE), BatchRecords(scales, RULE, fitted.tables),
                  BatchRecords(scales, RULE, fitted.tables, raw=True)]
    rows = []
    for video, index, _, base, symbols in samples(native, model, reader, plans["calibration"]):
        rows.extend(compare_sample(model, candidates, video, index, base, symbols))
    write_csv(args.output / "same_symbol.csv", rows)
    results = {}
    for budget in BUDGETS:
        group = [row for row in rows if row["budget"] == budget]
        sizes = {x: sum(row[f"{x}_bytes"] for row in group) for x in ("e0", "e1", "e2")}
        results[budget] = sizes | {f"{x}_savings_fraction": 1 - sizes[x] / sizes["e0"] for x in ("e1", "e2")}
    asset_file = args.output / "sensitivity.json"
    write_json(args.output / "summary.json", {
        "same_symbol": results, "matched_frames": len(rows) // len(BUDGETS), "read_counts": reader.counts,
        "static_cdf_hash": fitted.digest, "sensitivity_file_sha256": hashlib.sha256(asset_file.read_bytes()).hexdigest(),
        "all_records_e1_e2_pack_ms_not_compared_to_max_budget": True, "split": "Train holdout"})


if __name__ == "__main__":
    main()
