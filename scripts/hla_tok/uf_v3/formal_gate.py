"""Apply the two-node continuation rule without changing fitted coefficients."""

import argparse
from pathlib import Path

from hla_tok.evaluation.io import write_json
from scripts.hla_tok.v2.protocol import json_at


def score(row):
    return 10 * row["mse"] + .075 * row["lpips"] + .001 * row["actual_bpp"]


def decide(nodes):
    best, missed, evidence = None, 0, []
    for step, data in sorted(nodes.items()):
        cuts = {int(row["budget_mbps"]): row for row in data["quality"] if row["status"] == "PASS"}
        if {5, 10, 20, 40} - cuts.keys():
            return {"status": "STOP", "best_step": best[0] if best else step,
                    "reason": "a formal node lost a mandatory physical prefix", "nodes": evidence}
        composite = (score(cuts[5]) + score(cuts[10])) * .5
        severe = best is not None and any(
            cuts[b]["global_psnr"] < best[2][b]["global_psnr"] - .35 or cuts[b]["lpips"] > best[2][b]["lpips"] + .015
            for b in (20, 40))
        evidence.append({"step": step, "low_composite": composite, "high_rate_regression": severe})
        improved = not severe and (best is None or composite < best[1])
        if improved:
            best, missed = (step, composite, cuts), 0
        else:
            missed += 1
        if severe or missed >= 2:
            break
    stop = missed >= 2 or evidence[-1]["high_rate_regression"] or evidence[-1]["step"] >= 20_000
    if evidence[-1]["step"] == 8000 and best[0] == evidence[0]["step"]:
        stop = True
    return {"status": "STOP" if stop else "CONTINUE", "best_step": best[0],
            "nodes": evidence, "without_improvement": missed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    paths = sorted((args.results / "evaluation").glob("step-*/summary.json"))
    result = decide({int(p.parent.name.split("-")[-1]): json_at(p) for p in paths})
    write_json(args.results / "training/continuation.json", result)


if __name__ == "__main__":
    main()
