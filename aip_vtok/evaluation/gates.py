"""Pure, fail-closed gate decisions from complete measured artifacts."""

import math


def psnr(mse):
    return -10 * math.log10(max(mse, 1e-12))


def distortion_summary(rows):
    return {
        "frames": len(rows),
        "mse": sum(r["mse"] for r in rows) / len(rows),
        "psnr_global_mse": psnr(sum(r["mse"] for r in rows) / len(rows)),
        "psnr_frame_mean": sum(psnr(r["mse"]) for r in rows) / len(rows),
    }


def trial_passes(baseline, trial, limits):
    summary = trial["summary"]
    drop = baseline["psnr_global_mse"] - summary["psnr_global_mse"]
    return (
        summary["frames"] >= 64
        and trial["training_steps"] >= 300
        and math.isfinite(drop)
        and drop <= limits["maximum_ref_drop_db"]
        and trial["projector_latency"]["maximum_mean_ms"] < limits["maximum_projector_ms"]
    )


def assess_gate0(baseline, trials, limits):
    r0 = all(
        baseline[k]["frames"] >= 64 and baseline[k]["psnr_global_mse"] >= limits["minimum_r0_psnr_db"]
        for k in ("r0_native", "r0_aligned")
    )
    passing = [t for t in trials if trial_passes(baseline["r0_aligned"], t, limits)]
    return {
        "passed": bool(r0 and passing),
        "r0_passed": r0,
        "reference_passed": bool(passing),
        "next_gate_allowed": bool(r0 and passing),
        "limits": limits,
        "decision": "pass" if r0 and passing else "stop_at_gate0",
    }
