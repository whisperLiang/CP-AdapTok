"""Train, resume or validate the fixed-budget pilot after a full Phase 0 pass."""

import argparse
import csv
import json
import random
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2
import torch
import yaml

from hla_tok import HLATok
from hla_tok.evaluation.data import AuditedFrames, load_cohort
from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import FullFrameQuality, sha256_file
from hla_tok.evaluation.reconstruction import validate
from hla_tok.training.checkpoint import ResumeIdentity, read_checkpoint
from hla_tok.training.losses import Losses
from hla_tok.training.train import fixed_quality_gate, train


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", action="store_true")
    mode.add_argument("--resume", type=Path)
    mode.add_argument("--evaluate", type=Path)
    mode.add_argument("--benchmark", action="store_true")
    parser.add_argument("--bypass-vq", action="store_true", help="continuous-only evaluation of a fixed checkpoint")
    parser.add_argument("--stage", choices=("continuous", "fixed"), default="fixed")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/hla_tok/phase1.yaml"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("data/manifests"))
    parser.add_argument("--cohort", type=Path, default=Path("configs/hla_tok/validation_cohort.json"))
    parser.add_argument("--acceptance", type=Path, default=Path("results/hla_tok/acceptance.json"))
    parser.add_argument("--baseline", type=Path, default=Path("results/hla_tok/baselines/metrics.csv"))
    return parser.parse_args()


def require_pass(path, provenance):
    if not path.is_file():
        raise RuntimeError("Phase 0 evidence is missing")
    record = json.loads(path.read_bytes())
    if record["phase0"]["status"] != "PASS":
        raise RuntimeError(f"Phase 0 is {record['phase0']['status']}")
    if record["manifest_hashes"] != provenance["manifest_hashes"]:
        raise RuntimeError("Phase 0 and training manifest identities differ")


def verify_output(args):
    if args.resume is None and args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"output contains prior evidence: {args.output}")
    if args.train or args.resume is not None:
        if args.checkpoint_dir is None:
            raise ValueError("training requires a separate --checkpoint-dir")
        if args.stage == "fixed" and args.train and args.initialize_from is None:
            raise ValueError("start from a completed continuous diagnostic via --initialize-from")


def main():
    args = arguments()
    verify_output(args)
    if args.bypass_vq and (args.evaluate is None or args.stage != "fixed"):
        raise ValueError("--bypass-vq requires evaluation of a fixed VQ checkpoint")
    provenance = json.loads(args.cohort.read_bytes())
    reader = AuditedFrames(args.manifest_dir, provenance["manifest_hashes"])
    cohort = load_cohort(args.cohort, reader)
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cudnn.allow_tf32 = False
    try:
        if args.benchmark:
            from hla_tok.evaluation.benchmark import native_benchmark
            from hla_tok.evaluation.io import environment

            references = [(video, index) for video, count in cohort.items() for index in range(count)][:64]
            frames = [reader.read("validation", video, index) for video, index in references]
            env = environment()
            native_benchmark(torch.stack(frames[:64]).pin_memory(), args.output, env)
            return
        if args.evaluate is None:
            require_pass(args.acceptance, provenance)
        config = yaml.safe_load(args.config.read_bytes())
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        identity = ResumeIdentity(args.stage, sha256_file(args.config), sha256_file(args.cohort), commit)
        torch.manual_seed(config["seed"])
        random.seed(config["seed"])
        model = HLATok().cuda().to(memory_format=torch.channels_last)
        quality = FullFrameQuality("cuda:0", provenance["metric_weights"])
        if args.evaluate is not None:
            state = read_checkpoint(args.evaluate).state
            if (state["identity"]["dataset_sha256"] != identity.dataset_sha256
                    or state["identity"]["stage"] != args.stage):
                raise ValueError("checkpoint dataset identity or training stage differs")
            model.load_state_dict(state["model"])
            validate(model, reader, cohort, quality, (config["native_budget"],), args.output,
                     quantize=args.stage == "fixed" and not args.bypass_vq, bf16=config["precision"] == "bf16")
        else:
            losses = Losses("cuda:0", **config["loss"])
            train(model, reader, cohort, quality, losses, config, identity, args.output, args.checkpoint_dir,
                  resume=args.resume, initialize_from=args.initialize_from)
        table = args.output / "validation_metrics.csv"
        if table.is_file() and args.stage == "fixed" and not args.bypass_vq:
            with table.open() as stream:
                hla = next(csv.DictReader(stream))
            with args.baseline.open() as stream:
                original = list(csv.DictReader(stream))
            write_json(args.output / "fixed_quality_gate.json", fixed_quality_gate(
                {key: float(hla[key]) for key in ("actual_bpp", "global_psnr", "lpips", "dists")}, original))
    finally:
        if args.output.is_dir():
            path = args.output / "pixel_reads.json"
            older = json.loads(path.read_bytes()) if args.resume is not None and path.exists() else {}
            reads = {f"{k}_pixel_reads": v + older.get(f"{k}_pixel_reads", 0) for k, v in reader.counts.items()}
            write_json(path, reads)


if __name__ == "__main__":
    main()
