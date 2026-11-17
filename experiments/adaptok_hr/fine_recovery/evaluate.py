"""Matched Validation: actual packets, same-state prefixes and independent causal loops."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from adaptok_hr.codec.bitstream import HEADER, parse, truncate_packet
from adaptok_hr.codec.codec import Codec
from adaptok_hr.data.dataset import read_clip
from adaptok_hr.data.manifest import load_index
from adaptok_hr.evaluation.evaluate import (
    aggregate,
    evaluate_source,
    histogram_summary,
    metric_manifest,
    save_image,
)
from adaptok_hr.models.temporal_state import frame_context
from adaptok_hr.runtime import environment, file_hash, seed_everything, stable_hash, write_json
from adaptok_hr.training.checkpoint import load_model
from adaptok_hr.training.train import append_row
from cp_adaptok.metrics import MetricSuite

from .common import independent_codec
from .diagnose import LEVELS, detail_metrics, layer_slices

VALIDATION_IDS = ("DAVIS2017:blackswan", "DAVIS2017:car-roundabout", "DAVIS2017:dog")


def level_statistics(histograms):
    return {name: {**histogram_summary(counts.tolist()),
                   "maximum_code_fraction": float(counts.max() / max(counts.sum(), 1))}
            for name, counts in histograms.items()}


def empty_histograms():
    return {name: np.zeros(8192, dtype=np.int64) for name in LEVELS}


def count_levels(model, ids, histograms):
    for name, span in layer_slices(model, len(ids)):
        histograms[name] += np.bincount(ids[span], minlength=8192)


def aggregate_details(rows):
    result = aggregate(rows)
    for name in ("edge_mse", "high_frequency_l1", "high_frequency_mse"):
        if rows and name in rows[0]:
            result[name] = sum(r[name] for r in rows) / len(rows)
    return result


@torch.inference_mode()
def same_context(model, sources, frames, metrics, output):
    sender = Codec(model.eval(), amp_bf16=True)
    receiver = independent_codec(model)
    rows = {str(k): [] for k in model.config.budgets}
    histograms = {str(k): empty_histograms() for k in model.config.budgets}
    probes = []
    for source in sources:
        clip, _ = read_clip(source, 0, min(frames, source["original_frames"]))
        state = None
        for index, cpu in enumerate(clip):
            frame = cpu[None].cuda()
            full, next_state = sender.encode_frame(frame, state, 2048, source["content_id"])
            full_ids = parse(full).ids
            for budget in model.config.budgets:
                packet = truncate_packet(full, budget)
                parsed = parse(packet)
                if parsed.ids != full_ids[:budget]:
                    raise RuntimeError("IDs are not prefixes of a single encoding")
                rgb, decoded_state = receiver.decode_frame(packet, state)
                if budget == 2048 and not torch.equal(next_state.rgb, rgb):
                    raise RuntimeError("independent model receiver disagrees")
                row = {"content_id": source["content_id"], "frame": index,
                       "frame_type": decoded_state.frame_type, "tokens": budget, "pixels": 1080 * 1920,
                       "bytes": len(packet), "header_bytes": HEADER.size, "payload_bytes": len(packet) - HEADER.size,
                       "preceding_chain": state.chain.hex() if state is not None else None,
                       **metrics(frame, rgb.float()), **detail_metrics(rgb.float(), frame)}
                rows[str(budget)].append(row)
                count_levels(model, parsed.ids, histograms[str(budget)])
                append_row(output / "same_context.jsonl", row)
                if index in (0, 31, 32, 63):
                    directory = output / "same_context_packets" / source["content_id"].replace(":", "_")
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / f"frame_{index:04d}_K{budget}.bin").write_bytes(packet)
                if index in (0, 31, 63):
                    stem = f"{source['content_id'].replace(':', '_')}_{index:04d}"
                    save_image(output / "samples" / f"{stem}_K{budget}.png", rgb)
                    if budget == 2048:
                        save_image(output / "samples" / f"{stem}_original.png", frame)
            if index in (0, 7, 31, 63):
                probes.extend(fine_interventions(model, full_ids, state, frame, metrics, source["content_id"], index))
            state = next_state
    write_json(output / "fine_interventions.json", {"diagnostic_only": True, "frames": probes})
    return {k: {**aggregate_details(value), "levels": level_statistics(histograms[k])}
            for k, value in rows.items()}, probes


def fine_interventions(model, ids, state, frame, metrics, content_id, index):
    tokens = model.vq.lookup(torch.tensor([ids], dtype=torch.long, device="cuda"))
    offset = sum(model.positions.counts[:2])
    rng = torch.Generator().manual_seed(index + 17)
    permutation = torch.randperm(tokens.shape[1] - offset, generator=rng).cuda()
    shuffled = torch.cat((tokens[:, :offset], tokens[:, offset:][:, permutation]), 1)
    _, _, reference = frame_context(state, model.config.gop)
    rows = []
    for name, values in (("actual", tokens), ("shuffled_fine", shuffled), ("without_fine", tokens[:, :offset])):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rgb, _ = model.decoder(values, reference)
        rows.append({"content_id": content_id, "frame": index, "variant": name,
                     **metrics(frame, rgb.float()), **detail_metrics(rgb.float(), frame)})
    return rows


@torch.inference_mode()
def independent_loops(model, sources, frames, metrics, output):
    result = {}
    for budget in model.config.budgets:
        rows, histograms = [], empty_histograms()
        for source in sources:
            directory = output / f"closed_loop/k{budget}" / source["content_id"].replace(":", "_")
            _, measured = evaluate_source(model, source, budget, frames, metrics, output=directory)
            rows.extend(measured)
            for path in sorted(directory.glob("*.bin")):
                count_levels(model, parse(path.read_bytes()).ids, histograms)
        result[str(budget)] = {**aggregate(rows), "levels": level_statistics(histograms),
                               "early": aggregate([r for r in rows if r["frame"] < 8]),
                               "late": aggregate([r for r in rows if r["frame"] >= 48])}
        print(json.dumps({"budget": budget, "closed_loop": result[str(budget)]["psnr_global"]}), flush=True)
    return result


def run(args):
    root = Path(args.output)
    seed_everything(20261118)
    torch.set_num_threads(4)
    model, checkpoint = load_model(args.checkpoint)
    index = load_index(args.index)
    sources = [r for r in index["splits"]["validation"] if r["content_id"] in VALIDATION_IDS]
    if len(sources) != len(VALIDATION_IDS):
        raise ValueError("missing preregistered Validation source")
    protocol = {"checkpoint_sha256": file_hash(args.checkpoint), "manifest_identity": index["identity"],
                "frames_per_source": args.frames, "validation_ids": VALIDATION_IDS, "amp_bf16": True,
                "selection_split": "validation", "completed_steps": checkpoint["completed_steps"],
                "training_frames": checkpoint["training_frames"], "config": checkpoint["config"]}
    if (root / "run.json").exists():
        recorded = json.loads((root / "run.json").read_text())
        if recorded["identity"] != stable_hash(protocol):
            raise ValueError("evaluation directory belongs to a different protocol")
        if (root / "summary.json").exists():
            return
        # Keep interrupted evidence and replay the deterministic measurement into a fresh directory.
        attempt = len(list(root.parent.glob(root.name + ".interrupted-*"))) + 1
        root.rename(root.with_name(f"{root.name}.interrupted-{attempt:03d}"))
    root.mkdir(parents=True, exist_ok=True)
    write_json(root / "run.json", {"environment": environment(), "protocol": protocol, "identity": stable_hash(protocol)})
    metrics = MetricSuite("cuda", ("psnr", "lpips", "dists"))
    if metrics.errors:
        raise RuntimeError(metrics.errors)
    matched, probes = same_context(model, sources, args.frames, metrics, root)
    loops = independent_loops(model, sources, args.frames, metrics, root)
    fine = matched["2048"]["levels"]["Fine"]
    gain = matched["2048"]["psnr_global"] - matched["1024"]["psnr_global"]
    actual = [r["mse"] for r in probes if r["variant"] == "actual"]
    shuffled = [r["mse"] for r in probes if r["variant"] == "shuffled_fine"]
    information = sum(shuffled) / len(shuffled) - sum(actual) / len(actual)
    criteria = {"fine_multiple_ids": fine["used_codes"] > 1,
                "fine_shuffle_mse_increase": information, "fine_measurable_contribution": information > 1e-8,
                "same_context_gain_db": gain, "gain_at_least_0_1db": gain >= .1,
                "lpips_improves_with_budget": matched["2048"]["lpips"] < matched["1024"]["lpips"],
                "dists_improves_with_budget": matched["2048"]["dists"] < matched["1024"]["dists"],
                "closed_loop_gain_db": loops["2048"]["psnr_global"] - loops["1024"]["psnr_global"]}
    write_json(root / "summary.json", {"protocol": protocol, "same_context": matched,
                                       "same_encoding_prefix_ids_asserted_each_frame": True,
                                       "independent_model_receiver_asserted_each_frame": True,
                                       "closed_loop": loops, "criteria": criteria})
    write_json(root / "metric_manifest.json", metric_manifest())
    print(json.dumps(criteria), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--index", default="data/adaptok_hr/index.json")
    parser.add_argument("--frames", type=int, default=8)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
