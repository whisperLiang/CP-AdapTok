"""Time committed candidate checkpoints using the existing uninstrumented native codec."""

import argparse
import json
from statistics import median

from adaptok_hr.runtime import file_hash, write_json

from .run import ROOT, checkpoint, execute


def measure(name, step, repeats):
    path = checkpoint(name, step)
    for index in range(repeats):
        out = ROOT / name / f"benchmark_{step}_{index + 1}.json"
        budgets = (256, 512, 1024, 2048) if index == 0 else (2048,)
        if out.exists():
            saved = json.loads(out.read_text())
            if saved["checkpoint_sha256"] != file_hash(path):
                raise ValueError("throughput belongs to a different checkpoint")
            continue
        execute(["experiments.adaptok_hr.fine_recovery.benchmark", "--checkpoint", path,
                 "--output", out, "--budgets", *budgets], ROOT / "logs" / f"{name}_bench_{step}_{index + 1}.log")


def report():
    lines = ["## 单 H100 原生收发实测", "",
             "同一 car-roundabout 连续 64 帧，两个独立模型，包含发送端打包及本地解码、接收端解析及解码。"
             "排除读盘、H2D、指标和校验开销。每个包验证 RGB、参考特征和第 33 帧的独立重置。", "",
             "| 配置 | 步 | K | Encoder / Sender / Receiver FPS | 联合 FPS | P50 / P95 ms | 峰值 MiB | 参数量 | 64 帧字节 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    summaries = {}
    for file in sorted(ROOT.glob("S*/*benchmark_*_1.json")):
        data = json.loads(file.read_text())
        name = file.parent.name
        step = data["completed_steps"]
        for k, row in data["budgets"].items():
            t = row["timing"]
            lines.append(f"| {name} | {step} | {k} | {t['encoder']['fps']:.2f} / {t['sender']['fps']:.2f} / {t['receiver']['fps']:.2f} | "
                         f"{t['joint']['fps']:.2f} | {t['joint']['p50_ms']:.2f} / {t['joint']['p95_ms']:.2f} | "
                         f"{row['peak_gpu_mib']:.1f} | {data['parameters']} | {row['bytes']} |")
        repeats = [json.loads(p.read_text())["budgets"]["2048"]["timing"]["joint"]["fps"]
                   for p in sorted(file.parent.glob(f"benchmark_{step}_*.json"))]
        summaries[f"{name}/{step}"] = {"individual_joint_fps": repeats, "median_joint_fps": median(repeats)}
    write_json(ROOT / "benchmark_summary.json", summaries)
    (ROOT / "performance_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=[], help="Run name and step separated by a slash")
    parser.add_argument("--repeats", type=int, default=1)
    args = parser.parse_args()
    for run in args.runs:
        name, step = run.split("/")
        measure(name, int(step), args.repeats)
    report()
