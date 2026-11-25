"""Apply the frozen two-node formal continuation rule to physical Validation results."""

import argparse
import json
from pathlib import Path
from statistics import mean

from hla_tok.evaluation.io import write_json


def physical_summary(path: Path) -> dict[int, dict]:
    data = json.loads(path.read_text())
    rows = {int(row["budget_mbps"]): row for row in data["quality"] if row["budget_mbps"] in (5, 10, 20, 40)}
    if (set(rows) != {5, 10, 20, 40}
            or any(row["status"] != "PASS" or row["decoded"] != 174 for row in rows.values())):
        raise ValueError("training continuation requires all four complete 174-frame hard caps")
    return rows


def scalar(rows: dict[int, dict]) -> float:
    return mean(10 * rows[b]["mse"] + 0.075 * rows[b]["lpips"] + 0.001 * rows[b]["actual_bpp"] for b in (5, 10))


def evaluate(evidence: list[Path]) -> dict:
    best = physical_summary(evidence[0])
    best_value = scalar(best)
    stale = 0
    records = [{"path": str(evidence[0]), "low_rd": best_value, "improved": True}]
    status = "CONTINUE"
    for node in evidence[1:]:
        current = physical_summary(node)
        score = scalar(current)
        improved = score < best_value
        major_regression = any(current[b]["global_psnr"] < best[b]["global_psnr"] - 0.35
                               or current[b]["lpips"] > best[b]["lpips"] + .015 for b in (20, 40))
        stale = 0 if improved else stale + 1
        records.append({"path": str(node), "low_rd": score, "improved": improved,
                        "major_high_rate_regression": major_regression})
        if major_regression:
            status = "STOP_HIGH_RATE_REGRESSION"
            break
        if stale >= 2:
            status = "STOP_TWO_CONSECUTIVE_WITHOUT_GAIN"
            break
        if improved:
            best, best_value = current, score
    return {"status": status, "records": records, "consecutive_no_gain": stale}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summary", type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate(args.summary)
    write_json(args.output, result)
    print(result["status"])


if __name__ == "__main__":
    main()
