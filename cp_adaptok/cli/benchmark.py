import argparse

from cp_adaptok.cli.common import add_inference_options, inference_config, load_inference
from cp_adaptok.evaluation import benchmark_pipeline
from cp_adaptok.runtime import (
    MissingPrerequisites,
    metadata,
    report_missing,
    resolve,
    write_json,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="End-to-end actual-bitstream pipeline timing")
    add_inference_options(parser, "results/benchmark")
    parser.add_argument("--resolution", nargs=2, type=int, default=[1920, 1080], metavar=("WIDTH", "HEIGHT"))
    parser.add_argument("--measure", choices=("actual-bitstream",), required=True)
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--budgets", nargs="+", type=int)
    parser.add_argument("--reference-attention", action="store_true")
    args = parser.parse_args(argv)
    config = inference_config(args)
    try:
        model, status = load_inference(config, args, "benchmark")
        report = metadata(config, status)
        report["artifact"] = model.artifact_provenance
        report.update(
            benchmark_pipeline(
                model.eval(),
                config,
                args.resolution,
                args.frames,
                args.warmup,
                args.budgets or sorted({model.config.k_ref, model.config.n_max}),
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
