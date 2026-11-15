"""Run every required entry verbatim with the isolated Python; retain actual exit codes."""

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
commands = [
    ["-m", "pytest", "cp_adaptok/tests", "-q"],
    ["-m", "cp_adaptok.cli.train", "--config", "configs/s1_tokenizer.yaml", "--max-steps", "100"],
    [
        "-m",
        "cp_adaptok.cli.eval",
        "--config",
        "configs/s1_tokenizer.yaml",
        "--budgets",
        "128",
        "256",
        "512",
        "1024",
        "2048",
    ],
    ["-m", "cp_adaptok.cli.benchmark", "--resolution", "1920", "1080", "--measure", "actual-bitstream"],
    ["-m", "cp_adaptok.cli.stream_demo", "--mode", "localhost", "--progressive"],
]
output = ROOT / "results/required_commands"
output.mkdir(parents=True, exist_ok=True)
records = []
for i, args in enumerate(commands):
    start = time.perf_counter()
    with (output / f"{i}.log").open("w") as log:
        result = subprocess.run([sys.executable, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    records.append(
        {
            "argv": [sys.executable, *args],
            "exit_code": result.returncode,
            "elapsed_s": time.perf_counter() - start,
            "log": str(output / f"{i}.log"),
        }
    )
    print(json.dumps(records[-1]), flush=True)
(output / "commands.json").write_text(json.dumps(records, indent=2) + "\n")
raise SystemExit(0 if records[0]["exit_code"] == 0 and all(r["exit_code"] in (0, 2) for r in records) else 1)
