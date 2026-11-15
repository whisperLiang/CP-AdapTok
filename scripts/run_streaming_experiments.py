"""Run actual sequential TCP scenarios; retain commands, logs and observed outcomes."""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RATES = ["32", "128", "512", "2048", "128", "512", "2048", "512"]
SCENARIOS = {
    "all_prefixes_png": ["--max-frames", "2"],
    "tight_deadline": ["--allocation", "rate-only", "--playback-delay-ms", "300", "--bandwidth-kbps", *RATES],
    "fixed": ["--allocation", "fixed", "--fixed-k", "2048", "--bandwidth-kbps", *RATES],
    "rate_only": ["--allocation", "rate-only", "--bandwidth-kbps", *RATES],
    "scorer": ["--allocation", "scorer", "--bandwidth-kbps", *RATES],
    "truncated_tail": [
        "--allocation",
        "fixed",
        "--fixed-k",
        "2048",
        "--bandwidth-kbps",
        "2048",
        "--playback-delay-ms",
        "2500",
        "--enhancement-cap",
        "512",
    ],
    "delayed_ack": ["--allocation", "rate-only", "--bandwidth-kbps", "2048", "--ack-delay-ms", "100"],
}


def scenario_command(args, name, options):
    command = [
        sys.executable,
        "-m",
        "cp_adaptok.cli.stream_demo",
        "--mode",
        "localhost",
        "--progressive",
        "--config",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--output",
        str(args.output / name),
        "--playback-delay-ms",
        "1200",
        *options,
    ]
    if name != "all_prefixes_png":
        command.append("--no-save-frames")
    if args.allow_untrained:
        command.append("--allow-untrained")
    return command


def summarize(name, report):
    final = {row["frame"]: row for row in report["quality"]}
    return {
        "scenario": name,
        "status": report["status"],
        "frames": report["sample_count"],
        "delivered_tokens": [r["delivered_tokens"] for r in report["sender_frames"]],
        "selected_tokens": [r["selected_tokens"] for r in report["sender_frames"]],
        "application_bytes": report["total_application_bytes"],
        "omitted_enhancement_packets": report["omitted_enhancement_packets"],
        "base_wait_frames": report["base_wait_frames"],
        "deadline_missed_displays": report["deadline_missed_displays"],
        "mean_final_mse": sum(r["mse"] for r in final.values()) / len(final),
        "state_commits_match": report["sender_base_commits"] == report["receiver_base_commits"],
        "prefixes_and_state_checked": True,  # the CLI raises on byte, prefix or state mismatch
        "report": str(name + "/stream_demo.json"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", default="configs/streaming.yaml")
    parser.add_argument("--output", type=Path, default=ROOT / "results/streaming_followup")
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=list(SCENARIOS))
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    rows, commands = [], []
    for name in args.scenarios:
        command = scenario_command(args, name, SCENARIOS[name])
        with (args.output / f"{name}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        commands.append({"scenario": name, "argv": command, "exit_code": result.returncode})
        (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if result.returncode:
            raise RuntimeError(f"scenario {name} failed; see {args.output / (name + '.log')}")
        report = json.loads((args.output / name / "stream_demo.json").read_text())
        rows.append(summarize(name, report))
        print(json.dumps(rows[-1]), flush=True)
    summary = {
        "scenarios": rows,
        "conditions": "H100; each CLI has its own cold start; per-frame source clock; sequential runs",
        "limits": "application pacing, no TCP loss/congestion model; untrained results are not RD evidence",
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
