"""Common experiment artifacts and synchronized CUDA timings."""

import csv
import json
import subprocess
from pathlib import Path
from time import perf_counter

import torch


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty experiment table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def timed_cuda(call):
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda.synchronize()
    begin = perf_counter()
    start.record()
    result = call()
    stop.record()
    stop.synchronize()
    return result, (perf_counter() - begin) * 1000, start.elapsed_time(stop)


def environment() -> dict:
    driver = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True
    ).splitlines()[0]
    return {
        "gpu": torch.cuda.get_device_name(),
        "cuda": torch.version.cuda,
        "driver": driver,
        "torch": str(torch.__version__),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "attention": "PyTorch SDPA dispatcher",
        "compile": "eager",
    }
