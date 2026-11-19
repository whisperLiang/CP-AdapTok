"""Advance only the Train-selected arms, with independently locked thresholds."""

import math

LIMITS = {
    "a2_min_db": 0.3,
    "high_rate_min_db": 0.5,
    "phase1_lpips_max_increase": 0.005,
    "phase1_dists_max_increase": 0.005,
    "pilot_steps": 500,
    "pilot_same_min_db": 0.2,
    "pilot_closed_min_db": 0.0,
    "final_steps": 1000,
    "final_min_db": 0.5,
    "training_dists_max_increase": 0.005,
    "high_rate_lpips_weight": 10.0,
    "high_rate_dists_weight": 10.0,
    "high_rate_log2_time_weight": 0.05,
    "seed_order": [1, 2],
}
HISTORIES = ("same_context", "independent_closed")
SHORTLIST_ORDER = ("A2", "A3", "A5-fsq")
METHODS = ("B0-cont", "B0-quantized", "B1-quantized")


def deltas(baseline, arm):
    return {
        "psnr_gain_db": arm["psnr_global_mse"] - baseline["psnr_global_mse"],
        "lpips_delta": arm["lpips"] - baseline["lpips"],
        "dists_delta": arm["dists"] - baseline["dists"],
    }


def finite(row):
    return all(math.isfinite(row[m]) for m in ("psnr_global_mse", "lpips", "dists"))


def dense_gate(baseline, arm, name, limits):
    gain = deltas(baseline, arm)
    floor = limits["a2_min_db" if name == "A2" else "high_rate_min_db"]
    checks = {
        "finite": finite(baseline) and finite(arm),
        "psnr": gain["psnr_gain_db"] >= floor,
        "lpips": arm["lpips"] <= baseline["lpips"] + limits["phase1_lpips_max_increase"],
        "dists": arm["dists"] <= baseline["dists"] + limits["phase1_dists_max_increase"],
    }
    return {"passed": all(checks.values()), "checks": checks, **gain}


def time_ms(row):
    return max(row["encode_ms"] + row["decode_ms"], 1e-6)


def representative_score(baseline, row, limits):
    diff = deltas(baseline, row)
    return (
        diff["psnr_gain_db"] - limits["high_rate_lpips_weight"] * diff["lpips_delta"]
        - limits["high_rate_dists_weight"] * diff["dists_delta"]
        - limits["high_rate_log2_time_weight"] * math.log2(time_ms(row) / time_ms(baseline))
    )


def admit_on_train(rows, limits):
    eligible = [arm for arm in SHORTLIST_ORDER if dense_gate(rows["A1"], rows[arm], arm, limits)["passed"]]
    high = [arm for arm in eligible if arm != "A2"]
    chosen = [arm for arm in eligible if arm == "A2"]
    if high:
        chosen.append(max(high, key=lambda arm: representative_score(rows["A1"], rows[arm], limits)))
    return chosen


def release_dense(shortlist, validation, limits):
    if (any(arm not in SHORTLIST_ORDER for arm in shortlist)
            or len(set(shortlist)) != len(shortlist) or len(shortlist) > 2):
        raise ValueError("invalid frozen Train shortlist")
    candidates = {arm: dense_gate(validation["A1"], validation[arm], arm, limits) for arm in shortlist}
    return {"candidates": candidates, "released": [arm for arm, result in candidates.items() if result["passed"]]}


def history_gate(row, floor, limits):
    baseline, arm = (row[name] for name in METHODS[1:])
    gain = deltas(baseline, arm)
    checks = {
        "finite": finite(baseline) and finite(arm),
        "psnr": gain["psnr_gain_db"] >= floor,
        "lpips": arm["lpips"] < baseline["lpips"],
        "dists": arm["dists"] <= baseline["dists"] + limits["training_dists_max_increase"],
        "state_sync": row["state_sync"] is True,
    }
    return {"passed": all(checks.values()), "checks": checks, **gain}


def training_gate(table, milestone, limits):
    if milestone == limits["pilot_steps"]:
        floors = (limits["pilot_same_min_db"], limits["pilot_closed_min_db"])
    elif milestone == limits["final_steps"]:
        floors = (limits["final_min_db"],) * 2
    else:
        raise ValueError("gate requested at an unregistered milestone")
    histories = {name: history_gate(table[name], floor, limits) for name, floor in zip(HISTORIES, floors)}
    return {"passed": all(row["passed"] for row in histories.values()), "histories": histories}


def sparse_release(admission, seed1, seed2, integrity, limits):
    return (
        admission is True
        and all(record.get("step") == limits["final_steps"] and record.get("gate", {}).get("passed") is True
                and record.get("seed") == seed
                for seed, record in zip(limits["seed_order"], (seed1, seed2)))
        and integrity.get("passed") is True and integrity.get("test_images_read") == 0
    )
