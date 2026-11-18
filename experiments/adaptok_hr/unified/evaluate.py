"""Native Validation extends the verified packet evaluator with color and full-Fine controls."""

import argparse
import json
from dataclasses import replace

import torch

from adaptok_hr.codec.bitstream import parse, serialize, truncate_packet
from adaptok_hr.data.manifest import load_index
from adaptok_hr.evaluation.evaluate import aggregate, metric_manifest, save_image
from adaptok_hr.runtime import environment, file_hash, seed_everything, stable_hash, write_json
from adaptok_hr.training.checkpoint import load_model
from adaptok_hr.training.train import append_row
from cp_adaptok.metrics import MetricSuite
from experiments.adaptok_hr.fine_detail_v2.evaluate import (
    clips,
    closed_loop,
    same_context,
    suffix_variants,
)
from experiments.adaptok_hr.fine_recovery.common import independent_codec

from .run import PACKAGE, ROOT, checkpoint

PROTOCOL = json.loads((PACKAGE / "protocol.json").read_text())


def chroma(rgb):
    red, green, blue = rgb.float().unbind(1)
    luma = .2126 * red + .7152 * green + .0722 * blue
    return torch.stack(((blue - luma) / 1.8556, (red - luma) / 1.5748), 1)


def color_metrics(original, decoded):
    reference, actual = chroma(original), chroma(decoded)
    error = (reference - actual).abs().mean((0, 2, 3))
    target_variance = reference.var((-2, -1), unbiased=False).mean(0)
    variance = actual.var((-2, -1), unbiased=False).mean(0)
    return {"cb_mae": float(error[0]), "cr_mae": float(error[1]),
            "cb_variance": float(variance[0]), "cr_variance": float(variance[1]),
            "cb_target_variance": float(target_variance[0]), "cr_target_variance": float(target_variance[1])}


class NativeMetrics(MetricSuite):
    @torch.no_grad()
    def __call__(self, original, decoded):
        return {**super().__call__(original, decoded), **color_metrics(original, decoded)}


def summarize(rows):
    result = aggregate(rows)
    if not rows:
        return result
    for key in ("cb_mae", "cr_mae", "cb_variance", "cr_variance", "cb_target_variance", "cr_target_variance"):
        result[key] = sum(r[key] for r in rows) / len(rows)
    for channel in ("cb", "cr"):
        result[channel + "_variance_retention"] = result[channel + "_variance"] / max(result[channel + "_target_variance"], 1e-12)
    result["color_difference_variance_retention"] = ((result["cb_variance"] + result["cr_variance"]) /
                        max(result["cb_target_variance"] + result["cr_target_variance"], 1e-12))
    return result


def grouped_summary(rows):
    selectors = {"I": lambda r: r["frame_type"] == "I", "P": lambda r: r["frame_type"] == "P",
                 "late_GOP": lambda r: r["frame"] % 32 >= 24,
                 "late_sequence": lambda r: r["frame"] >= 48}
    result = {**summarize(rows), **{name: summarize(list(filter(predicate, rows)))
                                   for name, predicate in selectors.items()}}
    result["sources"] = {name: summarize([r for r in rows if r["content_id"] == name])
                         for name in sorted({r["content_id"] for r in rows})}
    return result


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def upgraded_summaries(path):
    rows = read_rows(path)
    return {str(k): grouped_summary([r for r in rows if r["tokens"] == k]) for k in (256, 512, 1024, 2048)}


def native_p_samples(receiver, data, state, frame, output, index):
    sample = output / "samples" / f"car-roundabout_{index:04d}"
    save_image(sample.with_name(sample.name + "_original.png"), frame)
    for k in (256, 512, 1024, 2048):
        decoded, _ = receiver.decode_frame(truncate_packet(data, k), state)
        save_image(sample.with_name(sample.name + f"_K{k}.png"), decoded)


def fine_variant(receiver, packet, ids, state, frame, actual, metadata, output, metrics):
    low, variant = metadata["fine_start"], metadata["variant"]
    if ids[:low] != packet.ids[:low]:
        raise RuntimeError("Fine intervention changed the Base/Medium IDs")
    changed = serialize(replace(packet, ids=ids))
    decoded, _ = receiver.decode_frame(changed, state)
    quality = metrics(frame, decoded.float())
    row = {**metadata, "actual_mse": actual["mse"], "mse_increase": quality["mse"] - actual["mse"],
           "lpips_increase": quality["lpips"] - actual["lpips"],
           "dists_increase": quality["dists"] - actual["dists"], **quality}
    append_row(output / "fine_interventions.jsonl", row)
    directory = output / "full_fine" / variant / metadata["content_id"].replace(":", "_")
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{metadata['frame']:04d}.bin").write_bytes(changed)
    return row


def fine_frame(receiver, data, donor, state, frame, actual, metadata, output, metrics):
    packet = parse(data)
    shuffle, cross = suffix_variants(packet.ids, donor, metadata["fine_start"], 913 + metadata["frame"])
    _, following = receiver.decode_frame(data, state)
    rows = []
    for variant, ids in (("shuffle", shuffle), ("cross", cross)):
        if ids is not None:
            info = {**metadata, "variant": variant, "donor": metadata["donor"] if variant == "cross" else None}
            rows.append(fine_variant(receiver, packet, ids, state, frame, actual, info, output, metrics))
    return following, rows


def intervention_summary(rows):
    result = {}
    for variant in ("shuffle", "cross"):
        selected = [r for r in rows if r["variant"] == variant]
        if selected:
            result[variant] = {"frames": len(selected), **{key: sum(r[key] for r in selected) / len(selected)
                        for key in ("mse_increase", "lpips_increase", "dists_increase")}}
    return result


def full_fine_controls(model, sources, frames, output, metrics, packets, canonical):
    receiver = independent_codec(model)
    low = sum(model.positions.counts[:2])
    names = [s["content_id"] for s in sources]
    common = min(frames, *(s["original_frames"] for s in sources))
    rows = []
    for content, clip in clips(sources, frames):
        state = None
        donor_name = names[(names.index(content) + 1) % len(names)]
        for index, cpu in enumerate(clip):
            frame, data = cpu[None].cuda(), packets[content, index]
            donor = parse(packets[donor_name, index]).ids if index < common else None
            if content == names[1] and index in (1, 31, 63):
                native_p_samples(receiver, data, state, frame, output, index)
            info = {"content_id": content, "frame": index, "fine_start": low, "donor": donor_name}
            state, measured = fine_frame(receiver, data, donor, state, frame, canonical[content, index], info, output, metrics)
            rows.extend(measured)
    return intervention_summary(rows)


def prepare_location(output, identity, sources, frames):
    if (output / "summary.json").exists():
        if json.loads((output / "summary.json").read_text())["identity"] != identity:
            raise ValueError("evaluation identity mismatch")
        return False
    if output.exists():
        attempt = len(list(output.parent.glob(output.name + ".interrupted-*"))) + 1
        output.rename(output.with_name(output.name + f".interrupted-{attempt}"))
    output.mkdir(parents=True)
    manifest = metric_manifest()
    manifest["chroma"] = "BT.709 full-range centered Cb/Cr; per-frame spatial variance; retention is sum decoded variances / sum source variances; denominator floor 1e-12"
    write_json(output / "run.json", {"identity": identity, "environment": environment(), "metrics": manifest,
             "requested_frames_per_source": frames,
             "actual_frames_per_source": {r["content_id"]: min(frames, r["original_frames"]) for r in sources},
             "64_frame_sources": [r["content_id"] for r in sources if r["original_frames"] >= 64]})
    return True


def evaluate(path, output, frames):
    seed_everything(20261201)
    torch.set_num_threads(4)
    model, saved = load_model(path)
    index = load_index(saved["config"]["data"]["index"])
    lookup = {r["content_id"]: r for r in index["splits"]["validation"]}
    sources = [lookup[s] for s in PROTOCOL["validation_sources"]]
    identity = {"checkpoint_sha256": file_hash(path), "manifest_identity": index["identity"],
                "protocol": stable_hash(PROTOCOL), "implementation": file_hash(__file__),
                "base_evaluator": file_hash(PACKAGE.parent / "fine_detail_v2/evaluate.py"), "frames": frames}
    if not prepare_location(output, identity, sources, frames):
        return
    metrics = NativeMetrics("cuda", ("psnr", "lpips", "dists"))
    if metrics.errors:
        raise RuntimeError(metrics.errors)
    with torch.inference_mode():
        _, packets, canonical = closed_loop(model, sources, frames, output, metrics)
        _, incremental, suffix = same_context(model, sources, frames, output, metrics, packets, canonical)
        fine = full_fine_controls(model, sources, frames, output, metrics, packets, canonical)
    loops = upgraded_summaries(output / "closed_loop.jsonl")
    matched = upgraded_summaries(output / "same_context.jsonl")
    write_json(output / "summary.json", {"identity": identity, "completed_steps": saved["completed_steps"],
              "training_frames": saved["training_frames"], "closed_loop": loops, "same_context": matched,
              "full_fine": fine, "incremental_fine": incremental, "suffix_statistics": suffix,
              "independent_endpoints_bitwise_equal": True, "nested_prefixes_verified": True,
              "same_state_gain_db": matched["2048"]["psnr_global"] - matched["1024"]["psnr_global"],
              "same_state_lpips_gain": matched["1024"]["lpips"] - matched["2048"]["lpips"],
              "parameters": sum(p.numel() for p in model.parameters())})
    print(json.dumps({"output": str(output), "K2048_LPIPS": loops["2048"]["lpips"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--frames", type=int, default=64)
    args = parser.parse_args()
    for name in args.names:
        evaluate(checkpoint(name, args.steps), ROOT / name / f"validation_{args.steps}_{args.frames}", args.frames)


if __name__ == "__main__":
    main()
