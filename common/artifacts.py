"""Common experiment artifacts and synchronized CUDA timings."""

import csv
import json
import subprocess
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("refusing to write an empty experiment table")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile("w", dir=path.parent, newline="", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    value = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    with NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(value)
            stream.flush()
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)




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
