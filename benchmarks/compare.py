"""Compare matched 1080p Validation receipts without treating token counts as rate."""

import argparse
from pathlib import Path
from statistics import mean

from common.artifacts import write_csv, write_json
from common.inputs import json_at
from common.metrics import psnr

RECEIPTS = ("ld", "hts", "pht", "d1", "adaptok")


def identity(artifact):
    meta = artifact["input"]
    frames = meta.get("frames") or meta["frame_hashes"]
    return meta.get("video", meta.get("content")), [item["sha256"] for item in frames]


def point(method, setting, metrics, byte_count=None, tokens=None):
    return {
        "method": method, "setting": str(setting), "frames": len(metrics),
        "psnr_global_db": psnr(mean(row["mse"] for row in metrics)),
        "psnr_mean_db": mean(row["psnr"] for row in metrics),
        "lpips": mean(row["lpips"] for row in metrics),
        "dists": mean(row["dists"] for row in metrics),
        "stream_bytes": byte_count,
        "bpp": byte_count * 8 / (len(metrics) * 1920 * 1080) if byte_count is not None else None,
        "tokens": tokens,
    }


def rate_distortion(receipts):
    rows = []
    for key in ("ld", "hts"):
        for setting in receipts[key]["checks"]:
            units = setting["units"]
            rows.append(point(f"dcvc_{key}", setting["qp"],
                              [metric for unit in units for metric in unit["quality"]],
                              sum(unit["bytes"] for unit in units)))
    fixed = receipts["pht"]["checks"]["fixed"]
    groups = {}
    for row in fixed:
        groups.setdefault((row["method"], row["stage"]), []).append(row)
    for (method, stage), units in groups.items():
        rows.append(point(f"pht/{method}", stage, [metric for unit in units for metric in unit["frame_metrics"]],
                          sum(unit["bytes"] for unit in units)))
    d1 = receipts["d1"]["checks"]
    for budget in ("5", "10", "20", "40"):
        points = [frame["points"][budget] for frame in d1]
        rows.append(point("d1", budget, [row["quality"] for row in points], sum(row["bytes"] for row in points)))
    for stride, values in receipts["adaptok"]["checks"].items():
        for budget, item in values.items():
            rows.append(point(f"adaptok/stride{stride}", budget, item["quality"], tokens=item["tokens"]))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="directory of smoke-*.py receipts named ld, hts, pht, d1, adaptok")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipts = {name: json_at(args.directory / f"{name}.json") for name in RECEIPTS}
    source = identity(receipts["ld"])
    for name, artifact in receipts.items():
        if identity(artifact) != source or artifact["input"]["test_pixel_reads"] != 0:
            raise ValueError(f"{name} does not use the same permitted Validation frames")
    write_csv(args.output / "rate_distortion.csv", rate_distortion(receipts))
    write_json(args.output / "codec_latency.json", {name: artifact["timing"] for name, artifact in receipts.items()})
    write_json(args.output / "network_streaming.json", receipts["pht"]["checks"]["network"])


if __name__ == "__main__":
    main()
