"""C0–C3 on one physical Base history per checkpoint and paired native source frame."""

import argparse
from collections import defaultdict
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.base import schedule
from hla_tok.uf.latent import LATENT_UNIT, normalized_error, render_latent
from hla_tok.uf.official import OfficialAssets, sha256_file
from hla_tok.uf.wire import MAX_FRAME_BYTES
from hla_tok.uf_v2.records import quantized_symbols
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v2.diagnostics import components, save_image, substitutions
from scripts.hla_tok.uf_v2.evaluate import allowance, summarize_timing, timing
from scripts.hla_tok.uf_v3.audit import checkpoints
from scripts.hla_tok.uf_v3.endpoint import load_v2, parse_base, standard_decoder
from scripts.hla_tok.v2.protocol import setup

CUTS = (0, 2, 5, 10, 20, 40, "ALL_VALID_RECORDS")
DIAGNOSTIC_TAG = b"capacity"


def native_format(latent):
    return latent.half().contiguous(memory_format=torch.channels_last)


def integer_stats(canvas, gathered, stats):
    counts = (canvas.levels >= 1, canvas.levels >= 2)
    active = stats[..., 1] != 0
    selected = sum(np.count_nonzero(plane[mask]) for plane, mask in zip(gathered, counts, strict=True))
    return {"level0": int(counts[0].sum()), "level1": int(counts[1].sum()),
            "nonzero_transmitted_symbols": int(selected), "nonzero_effective_symbols": int(
                np.count_nonzero(canvas.symbols)),
            "zero_coarse_dependency_records": int((counts[0] & ~active[0] & active[1]).sum()),
            "invalid_or_duplicate_records": int(sum((mask & ~valid).sum()
                for mask, valid in zip(counts, (active.any(axis=0), active[1]), strict=True)))}


def score_frame(codec, decoder, metric, clip, index, rgb, unreachable):
    intra, reset = schedule(index)
    base = codec.native.encode_base(rgb, index, codec.qp, intra, reset, codec.base_cap)
    original, enc_base, scale = codec.native.proxy(intra).ufv2_encode_snapshot()
    sym = quantized_symbols(normalized_error(original, enc_base, scale), codec.steps[intra])
    family = codec.bank.pack_symbols(base, DIAGNOSTIC_TAG, sym, MAX_FRAME_BYTES)
    decoded = decoder.shared_base(parse_base(family, DIAGNOSTIC_TAG))
    latent = decoder.decoded_latent()
    if not torch.equal(latent, enc_base) or not torch.equal(scale, decoded.scale):
        raise ValueError("encoder and independent Base decoder disagree")
    continuous = original.float() * scale.float()
    residual = continuous - latent.float()
    step = scale.float() * codec.steps[intra] * LATENT_UNIT
    canvas = codec.bank.canvas(tuple(sym.shape[1:]), intra)
    gathered, stats = canvas.geometry.native.gather(sym)
    support = int(np.count_nonzero(stats[1, :, 1]) + np.count_nonzero(stats[..., 1].any(axis=0)))
    if support != len(family.ends) - 1:
        raise ValueError("ALL_VALID_RECORDS did not exhaust the active dependency closure")
    metric.prepare(rgb)
    stem = {"video": clip.video, "index": index, "type": "I" if intra else "P", "base_bytes": family.base_end,
            "all_valid_bytes": len(family.raw), "all_valid_records": support}
    outputs = {"C0": render_latent(decoder, base.info, native_format(continuous), decoded.context),
               "C1": decoder.display(decoded, codec.refiner, residual)}
    rows = [stem | {"scenario": name, "budget": "CONTINUOUS", **metric.score(image)}
            for name, image in outputs.items()]
    previous = family.base_end
    scored = {}
    for budget in CUTS:
        cap = len(family.raw) if budget == "ALL_VALID_RECORDS" else allowance(budget) if budget else previous
        if cap < family.base_end or (not intra and budget in unreachable):
            status = "BASE_BUDGET_EXCEEDED" if cap < family.base_end else "DESYNC"
            unreachable.add(budget)
            rows.extend(stem | {"scenario": name, "budget": budget, "status": status}
                        for name in ("C2", "C3"))
            continue
        unreachable.discard(budget)
        raw, expected = family.prefix(cap)
        consumed, _ = canvas.decode(bytearray(raw[previous:]))
        if consumed != len(raw) - previous or canvas.records != expected:
            raise ValueError("physical and integer progression differ")
        previous = len(raw)
        delta = canvas.delta(step)
        if expected not in scored:
            if expected:
                a = render_latent(decoder, base.info, native_format(latent.float() + delta), decoded.context)
                b = decoder.display(decoded, codec.refiner, delta)
                scored[expected] = (a, b), (metric.score(a), metric.score(b))
            else:
                score = metric.score(decoded.rgb)
                scored[expected] = (decoded.rgb, decoded.rgb), (score, score)
        images, scores = scored[expected]
        counts = integer_stats(canvas, gathered, stats)
        if counts["invalid_or_duplicate_records"]:
            raise ValueError("native batch pack emitted an invalid record")
        lane = {"budget": budget, "bytes": len(raw), "records": expected, "unused_bytes": cap - len(raw),
                "wire_symbols": canvas.symbol_count, "latent_residual_mse": float((residual - delta).square().mean()),
                **counts}
        for name, image, score in zip(("C2", "C3"), images, scores, strict=True):
            outputs[f"{name}_{budget}"] = image
            rows.append(stem | lane | {"scenario": name, "status": "PASS", **score})
    return rows, outputs


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["scenario"], row["budget"])].append(row)
    result = {}
    for (scenario, budget), frames in grouped.items():
        complete = [row for row in frames if row.get("status", "PASS") == "PASS"]
        if len(complete) < len(frames):
            result[f"{scenario}/{budget}"] = {
                "status": "BASE_BUDGET_EXCEEDED", "frames": len(frames), "decoded": len(complete)}
            continue
        summary = summarize(frames)
        if "bytes" in frames[0]:
            integer_fields = ("bytes", "records", "level0", "level1", "nonzero_transmitted_symbols",
                              "nonzero_effective_symbols", "zero_coarse_dependency_records",
                              "invalid_or_duplicate_records", "wire_symbols")
            summary |= {key: sum(row[key] for row in frames) for key in integer_fields}
            summary |= {key: mean(row[key] for row in frames) for key in ("unused_bytes", "latent_residual_mse")}
            summary["actual_mbps"] = summary["bytes"] * 240 / len(frames) / 1e6
        result[f"{scenario}/{budget}"] = summary
    return result


def paired(rows):
    def take(name, budget):
        return {(r["video"], r["index"]): r for r in rows if (r["scenario"], r["budget"]) == (name, budget)}

    full = "ALL_VALID_RECORDS"
    comparisons = (("C0-C2", ("C0", "CONTINUOUS"), ("C2", full)),
                   ("C2-C3", ("C2", full), ("C3", full)),
                   ("C3-all-40", ("C3", full), ("C3", 40)))
    result = []
    for test, left, right in comparisons:
        a, b = take(*left), take(*right)
        if a.keys() != b.keys():
            raise ValueError("pairing keys disagree")
        for video, index in a:
            key = video, index
            result.append({"comparison": test, "video": video, "index": index,
                           **{name: a[key][name] - b[key][name] for name in ("psnr", "lpips", "dists")}})
    return result


def perception_rows(metric, rgb, decoded, video, index):
    trials = {"as_decoded": decoded} | substitutions(rgb, decoded)
    def edge(image):
        return (image[..., 1:, 1:] - image[..., :-1, 1:]).abs()
    return [{"video": video, "index": index, "intervention": name, **metric.score(image), **components(image, rgb),
             "vertical_edge_mae": (edge(image) - edge(rgb)).abs().mean().item()}
            for name, image in trials.items()]


def evaluate_seed(codec, reader, clips, out, limit, perception):
    metric, decoder = NativeQuality(), standard_decoder(codec)
    rows, perceptual, samples = [], [], []
    for clip in clips:
        codec.reset()
        decoder.reset()
        unreachable = set()
        for index in range(min(clip.length, limit)):
            rgb = source_frame(reader, clip, index)
            frame_rows, images = score_frame(codec, decoder, metric, clip, index, rgb, unreachable)
            rows.extend(frame_rows)
            if perception and index in (0, 1, 5, 11):
                perceptual.extend(perception_rows(metric, rgb, images["C3_ALL_VALID_RECORDS"], clip.video, index))
            if perception and index == 5 and clip == clips[0]:
                for name, image in {"source": rgb, **{key: images[key] for key in ("C0", "C1", "C2_40", "C3_40",
                                                                                  "C3_ALL_VALID_RECORDS")}}.items():
                    samples.append({"scenario": name, **save_image(out / "samples" / f"{name}.png", image)})
        print("capacity", clip.video, min(clip.length, limit), flush=True)
    write_csv(out / "frames.csv", rows)
    write_csv(out / "paired.csv", paired(rows))
    if perceptual:
        write_csv(out / "perceptual.csv", perceptual)
    if samples:
        write_json(out / "samples/manifest.json", {"video": clips[0].video, "index": 5, "images": samples})
    return {"pooled": aggregate(rows), "by_video": {clip.video: aggregate(
        [row for row in rows if row["video"] == clip.video]) for clip in clips}}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("calibration", "validation"), required=True)
    parser.add_argument("--limit", type=int, default=10000)
    parser.add_argument("--seed", type=int, choices=(1, 2), nargs="+", default=[1, 2])
    parser.add_argument("--d0-timing", action="store_true")
    args = parser.parse_args()
    setup()
    reader, plans = inputs()
    assets = OfficialAssets.verify(args.weights)
    clips = plans[args.split]
    data = {}
    for seed in args.seed:
        checkpoint = checkpoints(args.v2, seed)[20000]
        codec = load_v2(assets, checkpoint)
        data[seed] = evaluate_seed(codec, reader, clips, args.output / f"seed{seed}", args.limit, seed == 2)
        data[seed]["checkpoint_sha256"] = sha256_file(checkpoint)
        del codec
        torch.cuda.empty_cache()
    if args.d0_timing:
        codec = load_v2(assets, checkpoints(args.v2, 2)[20000], d0=True)
        budgets = (5, 10, 20, 40)
        rows = timing(codec, assets, reader, clips, budgets, 3, args.limit, False)
        write_csv(args.output / "d0_fast_timing.csv", rows)
        data["d0_fast"] = summarize_timing(rows, budgets, 3)
    write_json(args.output / "capacity.json", {"split": args.split, "results": data,
        "input_sha256": reader.hashes, "read_counts": reader.counts, "test_pixel_reads": reader.counts["test"],
        "history": "fixed within checkpoint and source frame; independent decoded Base and identical P context",
        "domain": "C0 uses normalized encoder y times native decoder-side q step; C1 is off-distribution control",
        "continuous_rate": None})


if __name__ == "__main__":
    main()
