"""Native-1080p paired and independent-loop evaluation, with no bitstream."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.nn import functional as F

from cp_adaptok.metrics import MetricSuite
from cp_adaptok.runtime import metadata, resolve, seed_everything, write_json
from experiments.fixed2048.checkpoint import load_reconstructor, sha256
from experiments.fixed2048.cli import read_config
from experiments.fixed2048.codec import PATHS, REFERENCES
from experiments.fixed2048.data import read_clip, records
from experiments.fixed2048.diagnostics import aggregate, content_sensitivity, distortion, token_statistics

FLAGS = {"real_bitstream": False, "bpp": None, "experiment_type": "reconstruction_diagnostic"}


def save_image(path, rgb):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    array = rgb.detach().float().squeeze(0).clamp(0, 1).mul(255).round().byte().permute(1, 2, 0).cpu().numpy()
    Image.fromarray(array).save(path)


def frame_metrics(model, encoded, output, rgb, suite):
    row = suite(rgb, output.rgb.float())
    row.update(
        distortion(encoded.latent, output.latent),
        latent_shape=list(encoded.latent.shape),
        pixels=rgb.shape[-2] * rgb.shape[-1],
        rgb_shape=list(output.rgb.shape),
    )
    if output.embeddings is not None:
        row["tokens"] = token_statistics(encoded.features, model.vq, output.ids)
    return row


def stage_times(encoded, output, path):
    times = dict(encoded.timings)
    if path in ("R0", "R0-spatial"):
        times.pop("tokenizer_ms", None)
    times.update(output.timings)
    times["end_to_end_ms"] = sum(times.values())
    return times


def paired_difference(outputs, rows, encoded, source, index, reference):
    by_path = {row["path"]: row for row in rows}
    return {
        "video": source["content_id"],
        "frame": index,
        "frame_type": encoded.context.frame_type,
        "reference": reference,
        "rgb_delta_mse": F.mse_loss(outputs["R1"].rgb, outputs["R2"].rgb).item(),
        "latent_delta_mse": F.mse_loss(outputs["R1"].latent, outputs["R2"].latent).item(),
        "r2_minus_r1_psnr": by_path["R2"]["psnr"] - by_path["R1"]["psnr"],
    }


def measure_frame(model, encoded, outputs, rgb, suite, source, index, reference, directory, last):
    rows = []
    for path, output in outputs.items():
        row = frame_metrics(model, encoded, output, rgb, suite)
        row.update(
            video=source["content_id"],
            frame=index,
            frame_type=encoded.context.frame_type,
            path=path,
            reference=reference,
            timings=stage_times(encoded, output, path),
            **FLAGS,
        )
        rows.append(row)
        if index in (0, last):
            save_image(
                directory / "images" / f"{Path(source['path']).name}_{reference}_{path}_{index:03}.png",
                output.rgb,
            )
    return rows


def diagnostic_content(model, encoded, previous, source, index, reference, directory):
    tests = content_sensitivity(model, encoded, previous)
    write_json(
        directory / "sensitivity" / f"{Path(source['path']).name}_{reference}_{index:03}.json",
        {
            "video": source["content_id"],
            "frame": index,
            "frame_type": encoded.context.frame_type,
            "diagnostic_only": True,
            "fixed_positions_and_context": True,
            "results": tests,
        },
    )


def advance_reference(model, encoded, outputs, paths, reference):
    policy = {
        "paired": ("R2", "original128"),
        "paired_full2048": ("R1", "full2048"),
        "matched128": (paths[0], "original128"),
    }
    path, mode = policy.get(reference, (paths[0], reference))
    return model.advance(encoded, outputs[path], mode)


def force_intra(reference, index):
    return reference == "i_only" or (reference in ("original128", "paired") and index % 32 == 0)


@torch.no_grad()
def evaluate_loop(model, row, clip, paths, reference, suite, output_dir, sensitivity=False):
    state, output_rows, paired = model.empty_state(), [], []
    previous_features = {}
    for index, frame in enumerate(clip):
        rgb = frame[None].to(model.device)
        encoded = model.encode(rgb, state, index, force_intra=force_intra(reference, index))
        outputs = model.compare(encoded, paths)
        measured = measure_frame(
            model, encoded, outputs, rgb, suite, row, index, reference, output_dir, len(clip) - 1
        )
        output_rows.extend(measured)
        if "R1" in outputs and "R2" in outputs:
            paired.append(paired_difference(outputs, measured, encoded, row, index, reference))
        frame_type = encoded.context.frame_type
        if sensitivity and frame_type in previous_features:
            diagnostic_content(
                model, encoded, previous_features[frame_type], row, index, reference, output_dir
            )
            sensitivity = False
        if encoded.features is not None:
            previous_features[frame_type] = encoded.features.detach()
        state = advance_reference(model, encoded, outputs, paths, reference)
        if index == 0:
            save_image(output_dir / "images" / f"{Path(row['path']).name}_original.png", rgb)
    return output_rows, paired


def write_tables(directory, rows):
    flat = [{key: value for key, value in row.items() if not isinstance(value, (dict, list))} for row in rows]
    keys = sorted(set().union(*(row.keys() for row in flat)))
    with (directory / "frame_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(flat)
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['reference']}/{row['path']}"].append(row)
    summary = {key: aggregate(values) for key, values in groups.items()}
    write_json(directory / "summary.json", {"groups": summary, **FLAGS})
    return summary


def check_test_seal(config, checkpoint, seal):
    if not seal:
        raise ValueError("test pixels require a frozen validation-selected experiment seal")
    saved = json.loads(resolve(seal).read_text())
    if sha256(checkpoint) not in saved["checkpoint_sha256s"]:
        raise ValueError("checkpoint identity is not present in the frozen test seal")
    if saved["test_config"] != config["final_test"]:
        raise ValueError("test configuration changed after sealing")
    if saved["test_manifest_sha256"] != sha256(resolve(config["manifest_dir"]) / "cpat_test.json"):
        raise ValueError("test manifest changed after sealing")
    return saved


def validate_test_request(config, seal, digest, paths, references, videos, frames):
    expected = {"paths": paths, "references": references}
    if seal["protocols"].get(digest) != expected:
        raise ValueError("test protocol differs from the frozen paths/reference modes")
    if videos != config["final_test"]["videos"] or frames != config["final_test"]["frames"]:
        raise ValueError("test protocol requires the declared complete video/frame selection")


def evaluate_sources(model, selected, paths, references, frames, suite, directory, sensitivity):
    rows, pairs = [], []
    with (directory / "frame_metrics.jsonl").open("w") as stream:
        for row in selected:
            clip = read_clip(row, frames)
            for reference in references:
                loops = (
                    [paths]
                    if reference in ("paired", "paired_full2048", "i_only")
                    else [[path] for path in paths]
                )
                for loop_paths in loops:
                    # I-only candidates share empty state; independent loops each start empty.
                    measured, paired = evaluate_loop(
                        model, row, clip, loop_paths, reference, suite, directory, sensitivity
                    )
                    rows.extend(measured)
                    pairs.extend(paired)
                    for item in measured:
                        stream.write(json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n")
                    stream.flush()
                    print(
                        json.dumps(
                            {
                                "video": row["content_id"],
                                "reference": reference,
                                "paths": loop_paths,
                                "frames": len(measured),
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
    return rows, pairs


def run_evaluation(
    config,
    checkpoint,
    output,
    paths,
    references,
    split="validation",
    videos=None,
    frames=8,
    sensitivity=False,
    seal=None,
):
    if split == "test":
        saved = check_test_seal(config, checkpoint, seal)
        validate_test_request(config, saved, sha256(checkpoint), paths, references, videos, frames)
    directory = resolve(output)
    if (directory / "run.json").exists():
        raise FileExistsError("evaluation output already exists; choose a new directory")
    directory.mkdir(parents=True, exist_ok=True)
    model, identity = load_reconstructor(checkpoint, config["device"], profile=True)
    suite = MetricSuite(model.device)
    if suite.errors:
        raise RuntimeError(f"required metrics unavailable: {suite.errors}")
    selected, hashes = records(config, split, videos)
    run = {
        "checkpoint": identity,
        "split": split,
        "manifest_hashes": hashes,
        "videos": [r["content_id"] for r in selected],
        "frames_per_video": frames,
        "paths": paths,
        "references": references,
        "evaluation_precision": "float32",
        "historical_intra_period": 32,
        "full2048_intra_policy": "first frame only",
        "latency": "synchronized component wall time; excludes metrics, file I/O and sensitivity",
        **FLAGS,
        **metadata(config, "running"),
    }
    write_json(directory / "run.json", run)
    rows, pairs = [], []
    torch.cuda.reset_peak_memory_stats(model.device)
    rows, pairs = evaluate_sources(model, selected, paths, references, frames, suite, directory, sensitivity)
    summary = write_tables(directory, rows)
    write_json(directory / "paired_quantization.json", pairs)
    run.update(
        status="complete",
        completed_frames=len(rows),
        peak_gpu_mib=torch.cuda.max_memory_allocated(model.device) / 2**20,
    )
    write_json(directory / "run.json", run)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fixed2048.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--paths", nargs="+", choices=PATHS, default=["R0", "R0-spatial", "R1", "R2"])
    parser.add_argument(
        "--references",
        nargs="+",
        choices=(*REFERENCES, "paired", "paired_full2048", "matched128"),
        default=["i_only", "paired", "full2048", "original128"],
    )
    parser.add_argument("--split", choices=["validation", "test"], default="validation")
    parser.add_argument("--videos", nargs="+")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--seal")
    args = parser.parse_args()
    config = read_config(args.config)
    seed_everything(config["seed"], config["device"], 0.9)
    videos = args.videos or config["final_test" if args.split == "test" else "baseline"]["videos"]
    run_evaluation(
        config,
        args.checkpoint or config["source_checkpoint"],
        args.output,
        args.paths,
        args.references,
        args.split,
        videos,
        args.frames,
        args.sensitivity,
        args.seal,
    )


if __name__ == "__main__":
    main()
