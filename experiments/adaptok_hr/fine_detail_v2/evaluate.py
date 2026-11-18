"""Decode every measured frame from packets; interventions never advance a reference."""

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from adaptok_hr.codec.bitstream import HEADER, parse, serialize, truncate_packet
from adaptok_hr.codec.codec import Codec
from adaptok_hr.data.dataset import read_clip
from adaptok_hr.data.manifest import load_index
from adaptok_hr.evaluation.evaluate import measured_quality, metric_manifest, save_image, verify_endpoints
from adaptok_hr.runtime import environment, file_hash, seed_everything, stable_hash, write_json
from adaptok_hr.training.checkpoint import load_model
from adaptok_hr.training.train import append_row
from cp_adaptok.metrics import MetricSuite
from experiments.adaptok_hr.fine_recovery.common import independent_codec
from experiments.adaptok_hr.fine_recovery.diagnose import detail_metrics
from experiments.adaptok_hr.fine_recovery.evaluate import aggregate_details, count_levels, empty_histograms, level_statistics

from .decisions import information
from .run import PROTOCOL, ROOT, checkpoint


def suffix_variants(ids, donor, low, seed):
    suffix = ids[low:]
    order = torch.randperm(len(suffix), generator=torch.Generator().manual_seed(seed)).tolist()
    shuffled = ids[:low] + tuple(suffix[i] for i in order)
    return shuffled, None if donor is None else ids[:low] + donor[low:]


def sha(data):
    return hashlib.sha256(data).hexdigest()


def folder(output, scope, budget, content):
    return output / scope / f"K{budget}" / content.replace(":", "_")


def summary(rows, hist):
    return {**aggregate_details(rows), "levels": level_statistics(hist),
            "sources": {s: aggregate_details([r for r in rows if r["content_id"] == s])
                        for s in sorted({r["content_id"] for r in rows})},
            "early": aggregate_details([r for r in rows if r["frame"] < 8]),
            "late": aggregate_details([r for r in rows if r["frame"] >= PROTOCOL["quality"]["late_start"]])}


def measured(frame, rgb, packet, content, index, original_hash, metrics):
    parsed = parse(packet)
    return {"content_id": content, "frame": index, "frame_type": "I" if parsed.intra else "P",
            "tokens": len(parsed.ids), "pixels": frame.shape[-1] * frame.shape[-2], "bytes": len(packet),
            "header_bytes": HEADER.size, "payload_bytes": len(packet) - HEADER.size,
            "input_rgb_sha256": original_hash, "packet_sha256": sha(packet), "previous": parsed.previous.hex(),
            **measured_quality(frame, rgb.float(), metrics), **detail_metrics(rgb.float(), frame)}


def clips(sources, frames):
    for source in sources:
        count = min(frames, source["original_frames"])
        clip, _ = read_clip(source, 0, count)
        yield source["content_id"], clip


def closed_loop(model, sources, frames, output, metrics):
    result, high_packets, high_rows = {}, {}, {}
    for budget in model.config.budgets:
        sender, receiver = Codec(model, True), independent_codec(model)
        rows, hist = [], empty_histograms()
        for content, clip in clips(sources, frames):
            sending, receiving = None, None
            directory = folder(output, "closed_loop", budget, content)
            directory.mkdir(parents=True, exist_ok=True)
            for index, cpu in enumerate(clip):
                frame = cpu[None].cuda()
                data, sending = sender.encode_frame(frame, sending, budget, content)
                rgb, receiving = receiver.decode_frame(data, receiving)
                verify_endpoints(sending, receiving)
                if index % model.config.gop == 0:
                    reset_rgb, reset = receiver.decode_frame(data, None)
                    verify_endpoints(sending, reset)
                row = measured(frame, rgb, data, content, index, sha(cpu.numpy().tobytes()), metrics)
                count_levels(model, parse(data).ids, hist)
                rows.append(row)
                append_row(output / "closed_loop.jsonl", row)
                (directory / f"{index:04d}.bin").write_bytes(data)
                if budget == model.config.budgets[-1]:
                    high_packets[content, index], high_rows[content, index] = data, row
        result[str(budget)] = summary(rows, hist)
        print(json.dumps({"loop": budget, "psnr": result[str(budget)]["psnr_global"]}), flush=True)
    return result, high_packets, high_rows


def suffix_diagnostic(receiver, packet, donor, state, frame, reference_mse):
    spec = PROTOCOL["information"]
    low, high = spec["suffix"]
    parsed = parse(packet)
    shuffled, swapped = suffix_variants(parsed.ids, donor, low, spec["scramble_seed"] + parsed.frame_index)
    suffix = parsed.ids[low:high]
    result = {"within_frame_used": len(set(suffix)), "actual_mse": reference_mse}
    for name, ids in (("shuffle", shuffled), ("cross", swapped)):
        mse = None
        if ids is not None:
            changed = serialize(replace(parsed, ids=ids))
            rgb, _ = receiver.decode_frame(changed, state)
            mse = float(F.mse_loss(rgb.float(), frame))
        result[name + "_mse"] = mse
        result[name + "_mse_increase"] = None if mse is None else mse - reference_mse
    return result


def same_context(model, sources, frames, output, metrics, packets, canonical):
    receiver = independent_codec(model)
    budgets = model.config.budgets
    rows = {k: [] for k in budgets}
    hists = {k: empty_histograms() for k in budgets}
    suffix_hist = np.zeros(8192, dtype=np.int64)
    probes = []
    low, high = PROTOCOL["information"]["suffix"]
    ids = [s["content_id"] for s in sources]
    common_frames = min(frames, *(s["original_frames"] for s in sources))
    for content, clip in clips(sources, frames):
        state = None
        donor_name = ids[(ids.index(content) + PROTOCOL["information"]["donor_rotation"]) % len(ids)]
        for index, cpu in enumerate(clip):
            frame = cpu[None].cuda()
            full = packets[content, index]
            full_ids = parse(full).ids
            original = canonical[content, index]
            for k in budgets:
                data = truncate_packet(full, k)
                parsed = parse(data)
                if parsed.ids != full_ids[:k]:
                    raise RuntimeError("packet prefix differs from the common encoding")
                rgb, next_state = receiver.decode_frame(data, state)
                if k == high:
                    if data != full:
                        raise RuntimeError("full packet replay is not exact")
                    row = original
                    following = next_state
                else:
                    row = measured(frame, rgb, data, content, index, original["input_rgb_sha256"], metrics)
                directory = folder(output, "same_context", k, content)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / f"{index:04d}.bin").write_bytes(data)
                if content == ids[1] and index in (0, 32):
                    sample = output / "samples" / f"car-roundabout_{index:04d}"
                    save_image(sample.with_name(sample.name + f"_K{k}.png"), rgb)
                    if k == high:
                        save_image(sample.with_name(sample.name + "_original.png"), frame)
                count_levels(model, parsed.ids, hists[k])
                rows[k].append(row)
                append_row(output / "same_context.jsonl", row)
            donor = parse(packets[donor_name, index]).ids if index < common_frames else None
            diagnostic = suffix_diagnostic(receiver, full, donor, state, frame, original["mse"])
            probe = {"content_id": content, "frame": index, "donor": donor_name if donor else None,
                     "diagnostic_only": True, **diagnostic}
            probes.append(probe)
            append_row(output / "suffix_interventions.jsonl", probe)
            suffix_hist += np.bincount(full_ids[low:high], minlength=8192)
            state = following
    return {str(k): summary(rows[k], hists[k]) for k in budgets}, information(probes), level_statistics({"suffix": suffix_hist})["suffix"]


def evaluate(name, steps, frames):
    seed_everything(20261118)
    torch.set_num_threads(4)
    path = checkpoint(name, steps)
    output = ROOT / name / f"validation_{steps}_{frames}"
    model, saved = load_model(path)
    index = load_index(saved["config"]["data"]["index"])
    lookup = {r["content_id"]: r for r in index["splits"]["validation"]}
    sources = [lookup[s] for s in PROTOCOL["sources"]]
    identity = {"checkpoint_sha256": file_hash(path), "manifest": index["identity"], "frames_per_source": frames,
                "protocol": stable_hash(PROTOCOL), "implementation": file_hash(__file__), "sources": PROTOCOL["sources"]}
    if (output / "summary.json").is_file():
        if json.loads((output / "summary.json").read_text())["identity"] != identity:
            raise ValueError("existing evaluation has a different identity")
        return
    if output.exists():
        attempt = len(list(output.parent.glob(output.name + ".interrupted-*"))) + 1
        output.rename(output.with_name(output.name + f".interrupted-{attempt}"))
    output.mkdir(parents=True)
    write_json(output / "run.json", {"identity": identity, "environment": environment(), "metrics": metric_manifest()})
    metrics = MetricSuite("cuda", ("psnr", "lpips", "dists"))
    if metrics.errors:
        raise RuntimeError(metrics.errors)
    with torch.inference_mode():
        loops, packets, canonical = closed_loop(model, sources, frames, output, metrics)
        matched, info, suffix = same_context(model, sources, frames, output, metrics, packets, canonical)
    write_json(output / "summary.json", {"identity": identity, "steps": saved["completed_steps"],
               "training_frames": saved["training_frames"], "closed_loop": loops, "same_context": matched,
               "information": info, "suffix": suffix, "same_state_gain_db": matched["2048"]["psnr_global"] - matched["1024"]["psnr_global"],
               "closed_loop_gain_db": loops["2048"]["psnr_global"] - loops["1024"]["psnr_global"],
               "all_closed_loop_endpoints_independent_and_equal": True, "all_prefixes_verified": True})
    print(json.dumps({"name": name, "information": info, "steps": steps}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--frames", type=int, required=True)
    args = parser.parse_args()
    for name in args.names:
        evaluate(name, args.steps, args.frames)


if __name__ == "__main__":
    main()
