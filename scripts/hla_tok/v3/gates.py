"""Preregistered quality and progression decisions on full native observations."""

from hla_tok import BUDGETS
from hla_tok.progressive.geometry import NATIVE_SHAPE
from hla_tok.streaming.fixed13 import TOKEN_BITS, payload_ledger
from hla_tok.training.train import fixed_quality_gate
from scripts.hla_tok.v3.quality import MAIN


def signed_gain(candidate, reference):
    return {key: (candidate[key] - reference[key]) * (1 if key == "global_psnr" else -1) for key in MAIN}


def quality_pass(row, thresholds):
    if row.get("frames") != 174 or not row.get("matched_174"):
        raise ValueError("only a full frozen native Validation result can admit a stage")
    return all(value >= 0 for value in signed_gain(row, thresholds).values())


def supports_extension(rows, thresholds, tolerances):
    if len(rows) < 2 or quality_pass(rows[-1], thresholds):
        return False
    if any(row["frames"] != 174 or not row["matched_174"] for row in rows[-2:]):
        return False
    return (all(delta > 0 for delta in signed_gain(rows[-1], rows[-2]).values())
            and all(delta >= -tolerances[key] for key, delta in signed_gain(rows[-1], thresholds).items()))


def preferred_alternative(candidate, control, config):
    gain = signed_gain(candidate, control)
    return (any(gain[key] >= limit for key, limit in config["meaningful_gain"].items())
            and all(gain[key] >= -limit for key, limit in config["material_degradation"].items()))


def discrete_thresholds(row, config, baselines):
    if row["mode"] != "fixed13" or row["rate_basis"] != "actual_fixed13_payload":
        raise ValueError("equal-rate admission requires a real Fixed13 payload")
    frames = row["frames"]
    tokens = config["native_budget"]
    total = ((tokens * TOKEN_BITS + 7) // 8) * frames
    reference = payload_ledger(total, tokens * frames, frames, *NATIVE_SHAPE)
    if (row["total_bytes"] == total and row["actual_bpp"] == reference["actual_bpp"]
            and round(reference["actual_bpp"], 6) == config["fixed_reference_bpp"]):
        return config["fixed_thresholds"]
    limits = fixed_quality_gate(row, baselines)["thresholds"]
    return {"global_psnr": limits["global_psnr_lower"], "lpips": limits["lpips_upper"], "dists": limits["dists_upper"]}


def progressive_decision(rows, config, baselines):
    if tuple(row["budget"] for row in rows) != BUDGETS:
        raise ValueError("progressive admission requires all eight ordered physical budgets")
    if any(row.get("frames") != 174 or not row.get("matched_174") or row["mode"] != "fixed13" for row in rows):
        raise ValueError("progressive admission requires full native Fixed13 observations")
    limits = config["progressive"]
    by_budget = {row["budget"]: row for row in rows}
    base, mid, fixed = (by_budget[budget] for budget in (BUDGETS[0], BUDGETS[2], config["native_budget"]))
    adjacent = [signed_gain(new, old) for old, new in zip(rows, rows[1:])]
    thresholds = discrete_thresholds(fixed, config, baselines)
    improvements = {
        "base_to_fixed_psnr": fixed["global_psnr"] - base["global_psnr"],
        "base_to_fixed_lpips": base["lpips"] - fixed["lpips"],
        "mid_to_fixed_psnr": fixed["global_psnr"] - mid["global_psnr"],
    }
    checks = {
        "fixed_quality": quality_pass(fixed, thresholds),
        "adjacent_average_monotonic": all(gain[key] >= -limits["tolerance"][key] for gain in adjacent for key in MAIN),
        "primary_gains": all(improvements[key] >= limit for key, limit in limits["gains"].items()),
        "not_saturated_in_top_three": any(gain[key] > limits["tolerance"][key] for gain in adjacent[-2:] for key in MAIN),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks,
            "fixed_thresholds": thresholds, "gains": improvements,
            "adjacent": [{"lower_budget": old, "upper_budget": new, **gain}
                         for old, new, gain in zip(BUDGETS, BUDGETS[1:], adjacent)]}
