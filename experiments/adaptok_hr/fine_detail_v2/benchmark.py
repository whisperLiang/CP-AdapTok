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
    triples = {name: row for name, row in summaries.items() if len(row["individual_joint_fps"]) == 3}
    if triples:
        lines.extend(["", "### K2048 同一 64 帧码流的三次联合吞吐", "",
                      "| 配置 / 步数 | 三次 FPS | 中位数 FPS |", "|---|---|---:|"])
        for name, row in triples.items():
            values = " / ".join(f"{x:.2f}" for x in row["individual_joint_fps"])
            lines.append(f"| {name} | {values} | {row['median_joint_fps']:.2f} |")
    write_json(ROOT / "benchmark_summary.json", summaries)
    (ROOT / "performance_report.md").write_text("\n".join(lines) + "\n")


def measure_validated():
    decision = json.loads((ROOT / "decision.json").read_text())
    repeated = {(decision["winner"], 1000)}
    decoder = ROOT / "decoder_decision.json"
    if decoder.exists():
        repeated.add((json.loads(decoder.read_text())["decoder_winner"], 1300))
    for path in sorted(ROOT.glob("S*/validation_*_*/summary.json")):
        name, step = path.parents[1].name, int(path.parent.name.split("_")[1])
        measure(name, step, 3 if (name.split("_", 1)[1], step) in repeated else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="*", default=[], help="Run name and step separated by a slash")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--all-validated", action="store_true", help="Time every evaluated checkpoint on an idle GPU")
    args = parser.parse_args()
    for run in args.runs:
        name, step = run.split("/")
        measure(name, int(step), args.repeats)
    if args.all_validated:
        measure_validated()
    report()
