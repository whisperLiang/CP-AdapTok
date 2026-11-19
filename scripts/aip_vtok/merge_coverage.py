"""Merge only traces of the final source; never use stale pre-refactor line numbers."""

from coverage import Coverage, CoverageData

from aip_vtok.runtime import ROOT, resolve, write_json


def main():
    names = (
        "tests_current",
        "engineering_final",
        "horizon_final",
        "replay_final",
        "diagnose_final",
        "benchmark_final",
        "gate1_final",
        "figures_final",
        "report_final",
        "audit_final",
        "trainer_final",
    )
    merged = CoverageData(basename=str(resolve("results/aip_vtok/coverage.final")))
    merged.erase()
    final_traces = {
        str(ROOT / "aip_vtok/decoder/latent_canvas.py"): {"tests_current"},
        str(ROOT / "aip_vtok/training/checkpoint.py"): {"tests_current", "trainer_final", "replay_final"},
        str(ROOT / "aip_vtok/training/trainer.py"): {"tests_current", "trainer_final", "replay_final"},
    }
    for name in names:
        source = CoverageData(basename=str(resolve(f"results/aip_vtok/coverage.{name}")))
        source.read()
        lines = {
            path: source.lines(path)
            for path in source.measured_files()
            if path not in final_traces or name in final_traces[path]
        }
        merged.add_lines(lines)
    merged.write()
    report = Coverage(data_file=str(resolve("results/aip_vtok/coverage.final")))
    report.load()
    report.json_report(outfile=str(resolve("results/aip_vtok/coverage.json")))
    write_json(
        "results/aip_vtok/coverage_sources.json",
        {
            "traces": names,
            "policy": "only final-source traces; old canvas records omitted after final no-grad guard",
        },
    )


if __name__ == "__main__":
    main()
