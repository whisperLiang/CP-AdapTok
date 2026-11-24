"""HLA-Tok V3 reconstruction-first experiment runner."""

import argparse
import fcntl
from pathlib import Path

from hla_tok import BUDGETS
from hla_tok.evaluation.metrics import FullFrameQuality, sha256_file
from scripts.hla_tok.v2.protocol import frozen_data, json_at, setup
from scripts.hla_tok.v3.benchmark import codec_benchmark, dense_benchmark
from scripts.hla_tok.v3.diagnostics import diagnose
from scripts.hla_tok.v3.downstream import (
    _approved_checkpoint,
    continuous_trials,
    full_pipeline,
    model_from_checkpoint,
    progressive_trials,
    quantized_trials,
    seed_two_r2,
)
from scripts.hla_tok.v3.experiments import dense_trials, legacy_model, on_gpu, require_pass, state_of
from scripts.hla_tok.v3.inputs import baseline_replay, preflight, r2_source
from scripts.hla_tok.v3.protocol import OUTPUT, PROTOCOL, ROOT, STAGES, read_config
from scripts.hla_tok.v3.quality import evaluate
from scripts.hla_tok.v3.samples import export_samples
from scripts.hla_tok.v3.sealed import evaluate_sealed, freeze_final
from scripts.hla_tok.v3.verify import verify_native

DEFAULT_V2 = ROOT.parent.parent / "hla-tok-v2" / ROOT.name


def _arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", type=Path, default=DEFAULT_V2)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("preflight")
    commands.add_parser("baseline-replay")
    for command in ("train", "resume"):
        child = commands.add_parser(command)
        child.add_argument("stage", choices=(*STAGES, "all"))
        child.add_argument("--seed", type=int, choices=(1, 2), default=1)
    for command in ("evaluate", "benchmark", "samples", "verify"):
        child = commands.add_parser(command)
        child.add_argument("--checkpoint", type=Path, required=command != "verify")
        child.add_argument("--output", type=Path)
        if command == "verify":
            child.add_argument("--mutations", action="store_true")
        if command == "evaluate":
            child.add_argument("--trend", action="store_true")
    child = commands.add_parser("diagnose")
    child.add_argument("--seed", type=int, choices=(1, 2), default=1)
    child = commands.add_parser("report")
    child.add_argument("--output", type=Path, default=OUTPUT)
    child.add_argument("--figures", action="store_true")
    commands.add_parser("freeze-final")
    commands.add_parser("sealed-test")
    return parser.parse_args()


def _train(args, config):
    if args.stage == "all":
        return full_pipeline(config, args.v2)
    if args.stage == "progressive":
        return progressive_trials(config, args.v2)
    if args.stage == "continuous":
        return continuous_trials(config, args.v2, args.seed)
    if args.stage == "quantized":
        return quantized_trials(config, args.v2, args.seed)
    if args.seed == 1:
        return dense_trials(config, args.v2)
    parent = seed_two_r2()
    if parent is None:
        return None
    variant = require_pass(state_of("dense", 1))["selected"]["variant"]
    return dense_trials(config, args.v2, 2, r2_override=parent, variants=(variant,))


def _open_model(args):
    if args.checkpoint is None:
        source, _ = r2_source(args.v2)
        return on_gpu(legacy_model(source)).eval(), source, "continuous", False
    model, state = model_from_checkpoint(args.checkpoint)
    stage = state["identity"]["stage"]
    mode = "dense" if stage.startswith("v3/dense/") else ("fixed13" if stage.endswith(("/C1", "/D")) else "continuous")
    return on_gpu(model).eval(), args.checkpoint, mode, stage.endswith("/D")


def _inference(args, config):
    setup()
    model, source, mode, progressive = _open_model(args)
    out = args.output or OUTPUT / args.command / sha256_file(source)[:16]
    reader, cohort, identity = frozen_data()
    fixed = config["native_budget"]
    budgets = BUDGETS if progressive else ((fixed,) if mode != "dense" else ())
    if args.command == "verify":
        if args.mutations:
            from scripts.hla_tok.v3.mutations import run_mutations

            run_mutations()
        if mode == "dense":
            raise ValueError("strict integer prefix verification requires R2")
        return verify_native(model, reader, cohort, fixed, out)
    if args.command == "benchmark":
        if mode == "dense":
            return dense_benchmark(model, reader, cohort, config, out)
        if mode != "fixed13":
            raise ValueError("the codec benchmark requires calibrated fixed or progressive VQ weights")
        return codec_benchmark(model, reader, cohort, config, out, BUDGETS)
    if args.command == "samples":
        return export_samples(model, reader, cohort, source, mode, out, budget=fixed)
    quality = FullFrameQuality("cuda", identity["metric_weights"])
    return evaluate(model, reader, cohort, quality, out, budgets=budgets, quantized=mode == "fixed13",
                    subset=args.trend, bypass=mode == "fixed13" and not progressive)


def _diagnose(args, config):
    setup(args.seed)
    decision = json_at(state_of("continuous", args.seed))
    if decision["candidate_quality"] != "FAIL":
        raise ValueError("B0 failure is required to open the conditional location diagnostic")
    dense_path = _approved_checkpoint(require_pass(state_of("dense", args.seed)))
    b0_folder = OUTPUT / "continuous" / f"seed{args.seed}" / "B0"
    b0_path = _approved_checkpoint({"selected": json_at(b0_folder / "latest.json")})
    dense, _ = model_from_checkpoint(dense_path)
    model, _ = model_from_checkpoint(b0_path)
    reader, cohort, identity = frozen_data()
    quality = FullFrameQuality("cuda", identity["metric_weights"])
    return diagnose(on_gpu(dense), on_gpu(model), reader, cohort, quality, config["native_budget"],
                    b0_folder / "bottleneck")


def _dispatch(args, config):
    if args.command == "preflight":
        path = OUTPUT / "preflight.json"
        return preflight(args.v2, path if not path.exists() else OUTPUT / "preflight_recheck.json")
    if args.command == "baseline-replay":
        output = OUTPUT / "baseline"
        if (output / "replay.json").is_file():
            result = require_pass(output / "replay.json")
            if result["source_sha256"] != sha256_file(r2_source(args.v2)[0]):
                raise ValueError("cached replay is not bound to the frozen R2 checkpoint")
            return result
        return baseline_replay(args.v2, output)
    if args.command in ("train", "resume"):
        return _train(args, config)
    if args.command == "freeze-final":
        return freeze_final(config)
    if args.command == "sealed-test":
        return evaluate_sealed(config)
    if args.command == "diagnose":
        return _diagnose(args, config)
    return _inference(args, config)


def main():
    args = _arguments()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    config = read_config(PROTOCOL)
    lock_name = ".report.lock" if args.command in ("report", "preflight") else ".runner.lock"
    with (OUTPUT / lock_name).open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.command == "report":
            from scripts.hla_tok.v3.report import report

            verdict = report(config, args.v2, args.output, figures=args.figures)
        else:
            verdict = _dispatch(args, config)
        if isinstance(verdict, dict) and "status" in verdict:
            print(verdict["status"])
        elif isinstance(verdict, list):
            print(f"observations: {len(verdict)}")


if __name__ == "__main__":
    main()
