"""Independent actual native LD rate points; no BD extrapolation."""

import argparse

from cp_adaptok.cli.eval import native_evaluation
from cp_adaptok.runtime import read_config, seed_everything, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qps", nargs="+", type=int, default=[16, 32, 48, 63])
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--output", default="results/optimization_20261115/native_controls")
    args = parser.parse_args()
    config = read_config("configs/pretrained_s1.yaml")
    seed_everything(config["seed"], config["device"], 0.8)
    summaries = []
    for qp in args.qps:
        config["model"]["qp"] = qp
        report = native_evaluation(config, config["device"], f"{args.output}/qp{qp}", max_frames=args.frames)
        records = report["records"]
        summaries.append(
            {
                "qp": qp,
                "bpp": sum(r["total_bytes"] for r in records) * 8 / (1920 * 1080 * len(records)),
                **{
                    k: sum(r[k] for r in records) / len(records)
                    for k in ["psnr", "mse", "lpips", "dists", "ms-ssim"]
                },
                "frames": len(records),
                "native_packet_framing": "SPS/IP",
                "same_rate_comparison": False,
            }
        )
    write_json(f"{args.output}/rd_summary.json", summaries)
    print(summaries)


if __name__ == "__main__":
    main()
