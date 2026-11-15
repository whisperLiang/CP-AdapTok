"""Serial final verification with exact commands, exit codes and fresh coverage."""

import json
import os
import subprocess
import sys
import time

from cp_adaptok.runtime import ROOT, read_config, write_json


def main():
    output = ROOT / "results/optimization_20261115/acceptance"
    output.mkdir(parents=True, exist_ok=True)
    config = read_config("configs/pretrained_s1.yaml")
    config["paths"].update(
        checkpoint="results/optimization_20261115/auxiliary/latest.pt",
        output=str(output / "training"),
        save_checkpoint=str(output / "training/latest.pt"),
    )
    config["training"].update(
        crop=256,
        sequence_length=2,
        tbptt=2,
        save_every=2,
        validate_every=2,
        diagnostic_every=1,
        budgets=[128, 2048],
        vq={"strategy": "stochastic_training", "temperature": 0.01},
    )
    config["data"]["validation"][0]["frames"] = 1
    config["evaluation"]["save_frames"] = False
    config["loss"]["dists"] = 0.01
    write_json(output / "config.json", config)
    prefix = [sys.executable, "-m"]
    checkpoint = "results/optimization_20261115/auxiliary/latest.pt"
    commands = [
        (
            "pytest",
            prefix
            + [
                "pytest",
                "cp_adaptok/tests",
                "-q",
                "--cov=cp_adaptok",
                f"--cov-report=json:{output}/coverage.json",
            ],
        ),
        (
            "train",
            prefix
            + [
                "coverage",
                "run",
                "--append",
                "-m",
                "cp_adaptok.cli.train",
                "--config",
                str(output / "config.json"),
                "--max-steps",
                "2",
            ],
        ),
        (
            "resume",
            prefix
            + [
                "coverage",
                "run",
                "--append",
                "-m",
                "cp_adaptok.cli.train",
                "--config",
                str(output / "config.json"),
                "--checkpoint",
                str(output / "training/latest.pt"),
                "--resume",
                "--max-steps",
                "3",
            ],
        ),
        (
            "eval",
            prefix
            + [
                "coverage",
                "run",
                "--append",
                "-m",
                "cp_adaptok.cli.eval",
                "--config",
                str(output / "config.json"),
                "--checkpoint",
                checkpoint,
                "--max-frames",
                "1",
                "--metrics",
                "psnr",
                "--output",
                str(output / "evaluation"),
            ],
        ),
        (
            "benchmark",
            prefix
            + [
                "coverage",
                "run",
                "--append",
                "-m",
                "cp_adaptok.cli.benchmark",
                "--config",
                str(output / "config.json"),
                "--checkpoint",
                checkpoint,
                "--measure",
                "actual-bitstream",
                "--frames",
                "2",
                "--warmup",
                "1",
                "--output",
                str(output / "benchmark"),
            ],
        ),
        ("coverage", prefix + ["coverage", "json", "-o", str(output / "coverage.json")]),
        (
            "audit",
            prefix
            + [
                "scripts.quality_audit",
                "--coverage",
                str(output / "coverage.json"),
                "--output",
                str(output / "quality_audit.json"),
            ],
        ),
        ("mutations", prefix + ["scripts.mutation_check", "--output", str(output / "mutations")]),
        ("lint", prefix + ["ruff", "check", "cp_adaptok", "scripts"]),
        ("whitespace", ["git", "diff", "--check"]),
    ]
    environment = dict(os.environ, COVERAGE_FILE=str(output / ".coverage"))
    environment["TRITON_LIBCUDA_PATH"] = str(ROOT / ".build/driver")
    records = []
    for name, command in commands:
        start = time.perf_counter()
        with (output / f"{name}.log").open("w") as log:
            result = subprocess.run(
                command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, timeout=600
            )
        records.append(
            {
                "name": name,
                "command": command,
                "exit_code": result.returncode,
                "elapsed_s": time.perf_counter() - start,
                "log": str(output / f"{name}.log"),
                "benchmark_instrumented_not_used_for_fps": name == "benchmark",
            }
        )
        write_json(output / "commands.json", records)
        print(json.dumps(records[-1]), flush=True)
    return int(any(row["exit_code"] for row in records))


if __name__ == "__main__":
    raise SystemExit(main())
