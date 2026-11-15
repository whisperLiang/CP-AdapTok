import argparse
import time

from cp_adaptok.data import read_video
from cp_adaptok.evaluation import evaluate_sequences
from cp_adaptok.metrics import MetricSuite
from cp_adaptok.runtime import (
    MissingPrerequisites,
    build_model,
    metadata,
    preflight,
    read_config,
    report_missing,
    resolve,
    seed_everything,
    write_json,
)
from cp_adaptok.streaming.sender import synchronize


def native_evaluation(config, device, output=None, split=None, metric_names=None, max_frames=None):
    from cp_adaptok.backbones.native_baseline import NativeBaseline

    paths = config["paths"]
    model = NativeBaseline(
        resolve(paths["ld_checkpoint"]), resolve(paths["intra_checkpoint"]), device, config["model"]["qp"]
    )
    suite = MetricSuite(device, metric_names or config["evaluation"]["metrics"])
    split = split or config["evaluation"]["split"]
    rows = []
    for configured_spec in config["data"][split]:
        spec = dict(configured_spec)
        if max_frames:
            spec["frames"] = min(spec["frames"], max_frames)
        model.reset()
        rows.extend(native_sequence(model, suite, spec, device))
    report = metadata(config, "native_pretrained")
    report.update(real_entropy_bitstream=True, split=split, records=rows, metric_errors=suite.errors)
    target = resolve(output or paths["output"]) / "native_baseline.json"
    write_json(target, report)
    print({"report": str(target), "frames": len(rows), "status": report["status"]})
    return report


def native_sequence(model, suite, spec, device):
    return [native_frame(model, suite, frame[None].to(device), spec) for frame in read_video(spec)]


def native_frame(model, suite, rgb, spec):
    synchronize(rgb.device)
    start = time.perf_counter()
    reconstruction, counts, _ = model.round_trip(rgb)
    synchronize(rgb.device)
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        **counts,
        **suite(rgb, reconstruction),
        "latency_ms": elapsed_ms,
        "source": spec["path"],
        "frame_type": "I" if model.frame_id == 1 else "P",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Actual bytes -> VQ IDs -> native-resolution reconstruction RD"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--budgets", nargs="+", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--split", choices=("test", "validation"))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--metrics", nargs="+")
    parser.add_argument("--output")
    parser.add_argument("--baseline", choices=("native",))
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.checkpoint:
        config["paths"]["checkpoint"] = args.checkpoint
    device = args.device or config["device"]
    output = resolve(args.output or config["paths"]["output"]) / "eval"
    try:
        preflight(config, "baseline" if args.baseline else "eval", args.allow_untrained)
        seed_everything(config["seed"], device, config["gpu_memory_fraction"])
        if args.baseline:
            native_evaluation(config, device, output, args.split, args.metrics, args.max_frames)
        else:
            model, status, _ = build_model(config, device, args.allow_untrained, args.checkpoint)
            report = evaluate_sequences(
                model.eval(),
                config,
                args.split or config["evaluation"]["split"],
                args.budgets,
                args.metrics,
                status,
                output,
                max_frames=args.max_frames,
            )
            print(
                {
                    "report": str(output / "evaluation.json"),
                    "status": status,
                    "budget_summary": report["budget_summary"],
                    "quality_drop_ratio": report["quality_drop_ratio"],
                }
            )
    except MissingPrerequisites as error:
        report_missing(output / "prerequisites.json", error, config)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
