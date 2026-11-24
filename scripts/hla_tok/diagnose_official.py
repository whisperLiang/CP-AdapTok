"""Probe equivalent alternatives to eager official AdapTok inference."""

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch

from experiments.original_adaptok_1080p.tiling import TileLayout, extract_batches
from hla_tok.evaluation.data import AuditedFrames, load_cohort
from hla_tok.evaluation.io import environment, timed_cuda, write_json
from hla_tok.evaluation.official import OfficialBaseline
from hla_tok.evaluation.official_cache import run_cache_diagnostics


@torch.inference_mode()
def compile_diagnostic(source, baseline, output):
    tiles = next(extract_batches(source.cuda().float().div_(255), TileLayout(1080, 1920), 1))[1]
    model = baseline.model

    def inference():
        ids, _ = baseline.encode(tiles)
        decoded = baseline.decode(ids[..., :256], 256)
        return ids, decoded

    with torch.autocast("cuda", dtype=torch.bfloat16):
        eager, eager_wall, eager_cuda = timed_cuda(inference)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            inference()
        kernels = sorted({event.key for event in profile.key_averages() if "scaled_dot_product" in event.key})
        write_json(output / "official_attention.json", {"precision": "bf16", "profile_ops": kernels})
        start = perf_counter()
        model.encoder = torch.compile(model.encoder, dynamic=False)
        model.decoder = torch.compile(model.decoder, dynamic=False)
        try:
            first, first_wall, _ = timed_cuda(inference)
            _, warm_wall, warm_cuda = timed_cuda(inference)
            error = (eager[1] - first[1]).abs()
            result = {
                "status": "MEASURED", "precision": "bf16", "tile_batch": 1,
                "integer_ids_equal": torch.equal(eager[0], first[0]),
                "rgb_exact": torch.equal(eager[1], first[1]), "rgb_max_abs": error.max().item(),
                "rgb_mean_abs": error.mean().item(), "eager_wall_ms": eager_wall, "eager_cuda_ms": eager_cuda,
                "first_call_wall_ms": first_wall, "compile_and_first_two_calls_s": perf_counter() - start,
                "warm_wall_ms": warm_wall, "warm_cuda_ms": warm_cuda,
            }
        except (RuntimeError, AssertionError, ImportError) as exc:
            result = {"status": "FAILED", "reason": str(exc), "elapsed_s": perf_counter() - start}
        write_json(output / "official_compile.json", result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cache", "compile"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--official-source", type=Path, required=True)
    parser.add_argument("--official-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    provenance = Path("configs/hla_tok/validation_cohort.json")
    identity = json.loads(provenance.read_bytes())
    reader = AuditedFrames(args.manifest_dir, identity["manifest_hashes"])
    cohort = load_cohort(provenance, reader)
    manifest = json.loads(Path("results/original_adaptok_1080p/checkpoint_manifest.json").read_bytes())
    try:
        weights = OfficialBaseline(args.official_source, args.official_checkpoint, manifest)
        video = next(iter(cohort))
        clip = torch.stack([reader.read("validation", video, index) for index in range(16)])
        write_json(args.output / "runtime.json", environment())
        if args.mode == "cache":
            run_cache_diagnostics(clip, weights, args.output)
        else:
            compile_diagnostic(clip, weights, args.output)
    finally:
        write_json(args.output / "pixel_reads.json", {f"{k}_pixel_reads": v for k, v in reader.counts.items()})


if __name__ == "__main__":
    main()
