import argparse

from cp_adaptok.evaluation import benchmark_pipeline
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


def main(argv=None):
    parser = argparse.ArgumentParser(description="End-to-end actual-bitstream pipeline timing")
    parser.add_argument("--config", default="configs/s1_tokenizer.yaml")
    parser.add_argument("--resolution", nargs=2, type=int, default=[1920, 1080], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--measure", choices=("actual-bitstream",), required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--budgets", nargs="+", type=int)
    parser.add_argument("--checkpoint")
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--device")
    parser.add_argument("--output", default="results/benchmark")
    parser.add_argument("--reference-attention", action="store_true")
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.checkpoint:
        config["paths"]["checkpoint"] = args.checkpoint
    device = args.device or config["device"]
    try:
        preflight(config, "benchmark", args.allow_untrained)
        seed_everything(config["seed"], device, config["gpu_memory_fraction"])
        model, status, _ = build_model(config, device, args.allow_untrained, args.checkpoint)
        report = metadata(config, status)
        report["artifact"] = model.artifact_provenance
        report.update(
            benchmark_pipeline(
                model.eval(),
                config,
                args.resolution,
                args.frames,
                args.warmup,
                args.budgets or [model.config.k_ref, model.config.n_max],
                not args.reference_attention,
            )
        )
        write_json(resolve(args.output) / "benchmark.json", report)
        print(
            {
                "status": status,
                "overall": report["overall"],
                "steady_inter": report["steady_inter"],
                "peak_allocated_bytes": report["peak_allocated_bytes"],
                "report": str(resolve(args.output) / "benchmark.json"),
            }
        )
    except MissingPrerequisites as error:
        report_missing(resolve(args.output) / "prerequisites.json", error, config)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
