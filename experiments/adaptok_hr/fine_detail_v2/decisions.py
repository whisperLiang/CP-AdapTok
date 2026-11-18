"""The preregistered selection rules operate solely on native Validation measurements."""

import json
from statistics import mean

from .run import PROTOCOL, ROOT, run_name


def information(rows):
    spec = PROTOCOL["information"]
    threshold = spec["minimum_mean_mse_increase_exclusive"]

    def summarize(values):
        cross = [r["cross_mse_increase"] for r in values if r["cross_mse_increase"] is not None]
        return {"shuffle": mean(r["shuffle_mse_increase"] for r in values),
                "cross": mean(cross) if cross else None,
                "multicode_frames": sum(r["within_frame_used"] > 1 for r in values),
                "frames": len(values), "donor_frames": len(cross)}

    per_source = {s: summarize([r for r in rows if r["content_id"] == s])
                  for s in sorted({r["content_id"] for r in rows})}
    pooled = summarize(rows)
    def passes(entry, floor):
        return (entry["multicode_frames"] > 0 and entry["shuffle"] > floor
                and entry["cross"] is not None and entry["cross"] > floor)
    agreeing = sum(passes(s, 0) for s in per_source.values())
    return {"pass": passes(pooled, threshold) and agreeing >= spec["minimum_sources_both_positive"],
            "agreeing_sources": agreeing, "pooled": pooled, "sources": per_source}


def result(name, steps):
    frames = PROTOCOL["screen"]["frames_per_source"] if steps == 300 else PROTOCOL["confirmation"]["frames_per_source"]
    return json.loads((ROOT / name / f"validation_{steps}_{frames}" / "summary.json").read_text())


def select():
    names = [arm for arm in PROTOCOL["arms"] if arm != "R00"]
    results = {arm: result(run_name(1, arm), 300) for arm in names}
    def rank(arm):
        row = results[arm]
        return (-int(row["information"]["pass"]), -row["same_state_gain_db"],
                row["closed_loop"]["2048"]["lpips"], arm)
    return sorted(names, key=rank)[0]


def deltas(candidate, reference):
    return {k: candidate[k] - reference[k] for k in ("psnr_global", "lpips", "dists")}


def quality(candidate, control, historical):
    rule = PROTOCOL["quality"]
    loops = candidate["closed_loop"]
    high, low = loops["2048"], loops["1024"]
    protected = {k: deltas(loops[k], control["closed_loop"][k]) for k in ("256", "512", "1024")}
    protected["late_2048"] = deltas(high["late"], control["closed_loop"]["2048"]["late"])
    absolute = {name: deltas(high, row["closed_loop"]["2048"])
                for name, row in (("control", control), ("old", historical))}
    checks = {"same_state_gain": candidate["same_state_gain_db"] >= rule["minimum_same_state_gain_db"],
              "closed_loop_gain": high["psnr_global"] > low["psnr_global"],
              "perception_with_budget": high["lpips"] < low["lpips"] and high["dists"] < low["dists"],
              "absolute_quality": all(d["psnr_global"] >= 0 and max(d["lpips"], d["dists"]) <= 0
                                      for d in absolute.values()),
              "protected": all(d["psnr_global"] >= -rule["protected_max_psnr_drop"]
                               and max(d["lpips"], d["dists"]) <= rule["protected_max_lpips_dists_increase"]
                               for d in protected.values()),
              "rate": high["bytes"] == rule["packet_bytes"] * high["frames"]}
    return {"pass": all(checks.values()), "checks": checks, "absolute_delta": absolute, "protected_delta": protected}
