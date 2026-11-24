"""Run isolated Phase 0 architecture or matched original model measurements."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import torch

from hla_tok.evaluation.data import AuditedFrames, load_cohort
from hla_tok.evaluation.io import environment, write_csv, write_json


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("benchmark", "pipeline", "baselines", "audit"))
    parser.add_argument("--cohort", type=Path, default=Path("configs/hla_tok/validation_cohort.json"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--official-source", type=Path)
    parser.add_argument("--official-checkpoint", type=Path)
    parser.add_argument("--joint-root", type=Path)
    parser.add_argument("--historical-dir", type=Path, default=Path("results/original_adaptok_1080p"))
    parser.add_argument("--frames", type=int, default=64, help="real input frames per HLA benchmark point")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--tile-batch", type=int, default=4)
    return parser.parse_args()


def read_benchmark_clip(reader, cohort, total):
    if not 1 <= total <= sum(cohort.values()):
        raise ValueError("benchmark frames must be within the matched cohort")
    frames = []
    for video, count in cohort.items():
        frames.extend(reader.read("validation", video, i) for i in range(min(count, total - len(frames))))
        if len(frames) == total:
            break
    return torch.stack(frames).pin_memory()


def run_baselines(args, reader, cohort, identity):
    from hla_tok.evaluation.joint_evidence import archived_joint_rows
    from hla_tok.evaluation.metrics import FullFrameQuality, sha256_file
    from hla_tok.evaluation.official import (
        OfficialBaseline,
        precision_diagnostics,
        quality_run,
        tile_batch_diagnostics,
    )

    if not (args.official_source and args.official_checkpoint and args.joint_root):
        raise ValueError("matched benchmarks require the official source, weights and Joint V1 provenance")
    manifest = json.loads((args.historical_dir / "checkpoint_manifest.json").read_bytes())
    model = OfficialBaseline(args.official_source, args.official_checkpoint, manifest)
    metrics = FullFrameQuality("cuda:0", identity["metric_weights"])
    rows = archived_joint_rows(args.joint_root, identity)
    write_csv(args.output / "joint_step40000_archived.csv", rows)
    for stride, precision in ((128, "fp32"), (128, "bf16"), (96, "fp32"), (96, "bf16")):
        target = args.output / f"stride{stride}_{precision}"
        rows += [row | {"measurement": "new_matched_full_frame"}
                 for row in quality_run(reader, cohort, model, metrics, stride, precision, args.tile_batch, target)]
        write_csv(args.output / "metrics.csv", rows)
    video = next(iter(cohort))
    clip = torch.stack([reader.read("validation", video, i) for i in range(16)])
    precision_diagnostics(clip, model, args.output, write_json)
    tile_batch_diagnostics(clip, model, args.output)
    after = sha256_file(args.official_checkpoint / "model.safetensors")
    if after != manifest["model_safetensors_sha256"]:
        raise RuntimeError("official checkpoint changed during the matched run")
    write_json(args.output / "official_weights.json", {
        "source": manifest["official_git_commit"], "revision": manifest["revision"],
        "weights_sha256_before": manifest["model_safetensors_sha256"], "weights_sha256_after": after,
        "metrics": metrics.weight_hashes,
    })


def main():
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"experiment output directory already contains evidence: {args.output}")
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    identity = json.loads(args.cohort.read_bytes())
    reader = AuditedFrames(args.manifest_dir, identity["manifest_hashes"])
    cohort = load_cohort(args.cohort, reader)
    historical = json.loads((args.historical_dir / "input_manifest.json").read_bytes())
    try:
        if args.operation == "audit":
            write_json(args.output / "cohort_audit.json", {
                "joint_commit": identity["joint_commit"], "frames": cohort,
                "manifest_hashes": reader.hashes,
                "historical_overlap": reader.historical_intersection(historical["sources"]),
            })
            return
        env = environment()
        write_json(args.output / "runtime.json", env)
        if args.operation == "benchmark":
            from hla_tok.evaluation.benchmark import native_benchmark

            native_benchmark(read_benchmark_clip(reader, cohort, args.frames), args.output, env,
                             repeats=args.repeats)
        elif args.operation == "pipeline":
            from hla_tok.evaluation.benchmark import native_pipeline

            references = [(video, i) for video, count in cohort.items() for i in range(count)][:args.frames]
            if len(references) != args.frames or args.frames < 1:
                raise ValueError("pipeline frames must be within the matched cohort")
            native_pipeline(reader, references, args.output, repeats=args.repeats)
        else:
            run_baselines(args, reader, cohort, identity)
    finally:
        write_json(args.output / "pixel_reads.json", {f"{split}_pixel_reads": count for split, count in reader.counts.items()})


if __name__ == "__main__":
    main()
