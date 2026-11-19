"""Paired S2 deltas and the predeclared recommendation without further training."""

from aip_vtok.c1.scan import BUDGETS

from .evaluation import CLOSED, FIXED
from .protocol import FOCUS, LIMITS
from .receiver import B2, B3


def point_index(points, method):
    return {(r["variant"], r["history"], r["budget"]): r["all"] for r in points if r["method"] == method}


def gains(before, after, variant):
    base, learned = point_index(before, B2), point_index(after, B3)
    for mode in (FIXED, CLOSED):
        for k in BUDGETS:
            a, b = base["base", mode, k], learned[variant, f"{mode}_B3", k]
            yield {"history": mode, "budget": k, "frames": b["frames"],
                   "psnr_gain_db": b["psnr_global_mse"] - a["psnr_global_mse"],
                   "mean_frame_psnr_gain_db": b["psnr_frame_mean"] - a["psnr_frame_mean"],
                   "lpips_change": b["lpips"] - a["lpips"], "dists_change": b["dists"] - a["dists"],
                   "baseline_rms": a["reference_latent_rms"], "rms": b["reference_latent_rms"]}


def paired_backslides(base, new, variant):
    def closed_psnr(rows, method, history, label):
        return {(r["lower"], r["upper"]): r for r in rows if r["method"] == method and r["history"] == history
                and r["metric"] == "psnr" and r["variant"] == label}

    a = closed_psnr(base, B2, CLOSED, "base")
    b = closed_psnr(new, B3, f"{CLOSED}_B3", variant)
    for lo, hi in zip(BUDGETS, BUDGETS[1:]):
        limit = max(LIMITS["max_psnr_backslide_floor_db"],
                    a[lo, hi]["max_backslide"] + LIMITS["max_psnr_backslide_over_b2_db"])
        yield {"lower": lo, "upper": hi, "limit_db": limit, "baseline_db": a[lo, hi]["max_backslide"],
               "observed_db": b[lo, hi]["max_backslide"], "passed": b[lo, hi]["max_backslide"] <= limit}


def focus_checks(deltas):
    focus = [r for r in deltas if r["history"] == CLOSED and r["budget"] in FOCUS]
    valid = len(focus) == len(FOCUS)
    passing = sum(r["psnr_gain_db"] >= LIMITS["focus_min_psnr_gain_db"] for r in focus)
    return {
        "focus_psnr": valid and passing >= LIMITS["focus_min_passing_budgets"]
        and all(r["psnr_gain_db"] > 0 for r in focus),
        "focus_lpips_dists": valid and all(meets_perceptual_gate(r) for r in focus),
    }


def meets_perceptual_gate(row):
    return row["lpips_change"] < 0 and row["dists_change"] <= LIMITS["focus_max_dists_increase"]


def horizon_checks(deltas, max_rms):
    late = [r for r in deltas if r["history"] == CLOSED]
    return {
        "late_quality": len(late) == len(BUDGETS) and all(
            min(r["psnr_gain_db"], r["mean_frame_psnr_gain_db"]) >= LIMITS["late_min_psnr_gain_db"]
            and r["lpips_change"] <= LIMITS["late_max_lpips_increase"]
            and r["dists_change"] <= LIMITS["late_max_dists_increase"] for r in late
        ),
        "reference_rms": max_rms <= LIMITS["max_reference_latent_rms"] and all(
            r["rms"] <= LIMITS["late_max_rms_relative_to_b2"] * r["baseline_rms"] for r in late
        ),
    }


def pilot_decision(baseline, result, variant, max_rms, integrity):
    scopes = ("validation", "horizon")
    deltas = {scope: list(gains(baseline[scope]["points"], result[scope]["points"], variant)) for scope in scopes}
    backslides = {scope: list(paired_backslides(baseline[scope]["backslides"], result[scope]["backslides"], variant))
                  for scope in scopes}
    deltas["late"] = list(gains(baseline["late"], result["late"], variant))
    checks = {
        "hard_integrity": integrity, **focus_checks(deltas["validation"]), **horizon_checks(deltas["late"], max_rms),
        "psnr_backslides": all(r["passed"] for chunk in backslides.values() for r in chunk),
    }
    return {"checks": checks, "passed": all(checks.values()), "max_reference_latent_rms": max_rms,
            "gains": deltas, "psnr_backslide_checks": backslides}
