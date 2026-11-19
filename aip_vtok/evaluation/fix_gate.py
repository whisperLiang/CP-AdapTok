"""Evaluate the Fix-A stop condition, Fix-B ablation and two-seed Fix-C release."""

import argparse
import json
from statistics import mean

import torch

from aip_vtok.evaluation.fix import frame_metrics, summarize_matched, summarize_validation
from aip_vtok.evaluation.fix_metrics import chroma_passes, pooled_fsq
from aip_vtok.evaluation.formal import summarize_interventions
from aip_vtok.evaluation.gates import distortion_summary
from aip_vtok.evaluation.smoke import BUDGETS
from aip_vtok.runtime import file_hash, resolve, write_json
from aip_vtok.training.fix import CONFIG, RESULTS, directories, fix_settings, read_json

PHASES = ("smoke", "fix_a", "fix_b", "fix_c")
HIGH = str(BUDGETS[-1])


def historical():
    diagnostics = read_json(RESULTS.parent / "gate1_seed1/diagnostics.json")["summary"][HIGH]
    horizon = read_json(RESULTS.parent / "gate1_horizon_diagnostics.json")["results"]
    return {
        "matched_psnr": diagnostics["psnr_global_mse"],
        "matched_direct_psnr": diagnostics["matched_direct"]["psnr_global_mse"],
        "raw_latent_mse": diagnostics["raw_latent_mse"],
        "fused_latent_mse": diagnostics["fused_latent_mse"],
        "horizon": {mode["mode"]: mode["summary"]["psnr_global_mse"] for mode in horizon},
    }


def matched_summary(directory):
    return summarize_matched(read_json(directory / "diagnostics.json")["rows"])


def horizon_summary(directory):
    evidence = read_json(directory / "horizon.json")
    rows = evidence["rows"]
    if evidence["gop"] != 32 or [row["frame"] for row in rows] != list(range(64)):
        raise ValueError("horizon evidence must contain one uninterrupted GOP32 64-frame rollout")
    late = rows[48:64]
    return {
        "all": distortion_summary(rows),
        "early_1_3": distortion_summary(rows[1:4]),
        "late_48_63": distortion_summary(late),
        "late_native_feature_l1": mean(row["reference_feature_l1"] for row in late),
        "late_max_reference_rms": max(row["reference_latent_rms"] for row in late),
        "endpoints": {str(frame): rows[frame] for frame in (31, 63)},
    }


def reference_stable(horizon, limits):
    ceiling = limits["maximum_reference_latent_rms"]
    for row in horizon["endpoints"].values():
        rms, native = row["reference_latent_rms"], row["native_aligned_latent_rms"]
        if not (0 < rms < ceiling and native > 0):
            return False
        if (
            not limits["minimum_reference_to_native_rms"]
            <= rms / native
            <= limits["maximum_reference_to_native_rms"]
        ):
            return False
    return horizon["late_max_reference_rms"] < ceiling


def verified_evaluation(directory, *, final=True):
    result = read_json(directory / "evaluation.json")
    if final:
        training = read_json(directory / "training_complete.json")
        if (result["step"], result["checkpoint_sha256"]) != (
            training["steps"],
            training["checkpoint_sha256"],
        ):
            raise ValueError("evaluation and training attestations differ")
    return result


def assess_smoke(directory, config):
    limits = config["fix_limits"]
    evaluation = verified_evaluation(directory)
    history = read_json(directory / "training.json")["history"]
    diagnostics = matched_summary(directory)[HIGH]
    local = read_json(directory / "overfit.json")
    losses = [mean(row["loss"] for row in selected) for selected in (history[:10], history[-10:])]
    clip_psnr = [
        distortion_summary([row for row in local[label] if row["budget"] == int(HIGH)])["psnr_global_mse"]
        for label in ("initial", "final")
    ]
    gradients = [row["records"][0].get("weighted_grad_norm_at_latent", {}) for row in history]
    checks = {
        "200_steps": evaluation["step"] == len(history) == 200,
        "closed_loop_stable": reference_stable(horizon_summary(directory), limits),
        "training_clip_improved": losses[1] < losses[0] and clip_psnr[1] > clip_psnr[0],
        "frozen_projector_gradient_to_prefix": any(g.get("reference", 0) > 0 for g in gradients)
        and all(row["module_gradients"]["projector"] == 0 for row in history),
    }
    anchor_checks = {
        "fusion_delta_small": diagnostics["delta_fusion_mse"] <= limits["maximum_fusion_delta_mse"],
        "fused_mse_reduced": diagnostics["fused_latent_mse"]
        <= limits["maximum_fix_a_fused_mse_fraction"] * historical()["fused_latent_mse"],
    }
    return {
        "passed": all(checks.values()) and all(anchor_checks.values()),
        "checks": checks,
        "matched_anchor_checks": anchor_checks,
        "loss_first_last10": losses,
        "overfit_k8192_psnr_initial_final": clip_psnr,
    }


def assess_fix_a(directory, config):
    limits, baseline = config["fix_limits"], historical()
    evaluation = verified_evaluation(directory)
    diagnosis = matched_summary(directory)[HIGH]
    measured = summarize_validation(frame_metrics(directory))["paired"]
    psnr_gain = diagnosis["psnr_global_mse"] - baseline["matched_psnr"]
    mse_ratio = diagnosis["fused_latent_mse"] / baseline["fused_latent_mse"]
    checks = {
        "1000_steps": evaluation["step"] == 1000,
        "validation_174_frames": all(row["frames"] == 174 for row in measured.values()),
        "closed_loop_stable": reference_stable(horizon_summary(directory), limits),
        "matched_psnr_improves": diagnosis["frames"] == 6
        and psnr_gain >= limits["minimum_fix_a_psnr_gain_db"],
        "matched_fused_mse_improves": mse_ratio <= limits["maximum_fix_a_fused_mse_fraction"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "fusion_delta_small": diagnosis["delta_fusion_mse"] <= limits["maximum_fusion_delta_mse"],
        "matched_psnr_gain_db": psnr_gain,
        "matched_fused_mse_fraction": mse_ratio,
    }


def exact_independent(directory):
    rows = read_json(directory / "independent_exactness.json")
    return len(rows) == 64 * len(BUDGETS) and all(
        row["max_rgb_error"] == 0 and row["equal_ids"] and row["equal_reference"] for row in rows
    )


def complete_modes(measures):
    return all(
        len(measures.get(mode, {})) == len(BUDGETS) and all(r["frames"] == n for r in measures[mode].values())
        for mode, n in (("paired", 174), ("independent_64", 64), ("late_48_63", 16))
    )


def budget_sequence(training):
    return [[row["budget"] for row in step["records"]] for step in training["history"]]


def fork_evidence():
    directories_and_weights = [directories(branch, 1) for branch in ("b0", "b1")]
    b0, b1 = [
        torch.load(weights / "latest.pt", map_location="cpu", weights_only=False)
        for _, weights in directories_and_weights
    ]
    baseline = torch.load(directories("fix_a", 1)[1] / "latest.pt", map_location="cpu", weights_only=False)
    inputs = [read_json(directory / "inputs.json") for directory, _ in directories_and_weights]
    projector_path = resolve("checkpoints/aip_vtok/projector_w128/latest.pt")
    original = torch.load(projector_path, map_location="cpu", weights_only=False)["projector"]
    fork_step = b0["config"]["stage1"]["fork_step"]
    fork = directories("fix_a", 1)[1] / f"step_{fork_step:04d}.pt"
    groups = b1["optimizer"]["param_groups"]
    return {
        "same_fork": b0["hashes"]["fork_checkpoint"] == b1["hashes"]["fork_checkpoint"] == file_hash(fork),
        "same_data_and_augmentation_plans": inputs[0]["plans"] == inputs[1]["plans"],
        "same_sampled_budgets": budget_sequence(b0) == budget_sequence(b1),
        "same_final_four_rng_streams": all(
            torch.equal(v, b1["streams"][k]) for k, v in b0["streams"].items()
        ),
        "b0_replays_fix_a": all(
            torch.equal(value, baseline["model"][key]) for key, value in b0["model"].items()
        ),
        "projector_group_lr": len(groups) == 2 and groups[1]["lr"] == 0.1 * groups[0]["lr"],
        "joint_projector_stored_and_updated": "projector" in b1
        and b1["projector"].keys() == original.keys()
        and any(not torch.equal(b1["projector"][key], value) for key, value in original.items()),
        "frozen_projector_artifact_omitted": "projector" not in b0,
    }


def wins_metrics(candidate, baseline):
    return (
        candidate["psnr_global_mse"] > baseline["psnr_global_mse"]
        and candidate["psnr_frame_mean"] > baseline["psnr_frame_mean"]
        and candidate["lpips"] < baseline["lpips"]
        and candidate["dists"] < baseline["dists"]
    )


def assess_fix_b(config):
    dirs = [directories(branch, 1)[0] for branch in ("b0", "b1")]
    b0, b1 = [summarize_validation(frame_metrics(directory)) for directory in dirs]
    h0, h1 = [horizon_summary(directory) for directory in dirs]
    checks = {
        "500_steps_both": all(verified_evaluation(d)["step"] == 500 for d in dirs),
        "complete_measured_sets": complete_modes(b0) and complete_modes(b1),
        "exact_independent": all(exact_independent(directory) for directory in dirs),
        "b0_reference_stable": reference_stable(h0, config["fix_limits"]),
        **read_json(RESULTS / "fix_b/fork_audit.json"),
    }
    wins = {
        mode: all(wins_metrics(b1[mode][str(budget)], b0[mode][str(budget)]) for budget in BUDGETS)
        for mode in ("paired", "independent_64")
    }
    wins.update(
        late_k8192=wins_metrics(b1["late_48_63"][HIGH], b0["late_48_63"][HIGH]),
        late_reference_l1=h1["late_native_feature_l1"] < h0["late_native_feature_l1"],
        reference_stable=reference_stable(h1, config["fix_limits"]),
    )
    passed = all(checks.values())
    return {
        "passed": passed,
        "checks": checks,
        "b1_wins": wins,
        "winner": ("b1" if all(wins.values()) else "b0") if passed else None,
    }


def progresses(rows, minimum_core_gain):
    for lower, upper in zip(BUDGETS, BUDGETS[1:]):
        a, b = rows[str(lower)], rows[str(upper)]
        if not wins_metrics(b, a):
            return False
        if lower in (1024, 2048) and b["psnr_global_mse"] - a["psnr_global_mse"] < minimum_core_gain:
            return False
    return True


def healthy_fsq(pooled, limits):
    return set(pooled) == {"0", "1", "2"} and all(
        row["pooled"]["tanh_saturation_ratio"] <= limits["maximum_tanh_saturation_ratio"]
        and min(row["pooled"]["dim_entropy_bits"]) > limits["minimum_pooled_dim_entropy_bits"]
        for row in pooled.values()
    )


def assess_seed(directory, config):
    limits = config["fix_limits"]
    evaluation = verified_evaluation(directory)
    validation = summarize_validation(frame_metrics(directory))
    paired = validation["paired"]
    diagnosis = matched_summary(directory)[HIGH]
    interventions = summarize_interventions(read_json(directory / "interventions.json"))
    health = pooled_fsq(read_json(directory / "health.json"))
    checks = {
        "formal_training": evaluation["step"] == config["stage1"]["steps"],
        "complete_modes": complete_modes(validation),
        "exact_independent_64": exact_independent(directory),
        "closed_loop_stable": reference_stable(horizon_summary(directory), limits),
        "matched_latent_alignment": diagnosis["frames"] == 6
        and diagnosis["delta_fusion_mse"] <= limits["maximum_fusion_delta_mse"]
        and diagnosis["fused_latent_mse"]
        <= limits["maximum_fix_a_fused_mse_fraction"] * historical()["fused_latent_mse"],
        "healthy_fsq": healthy_fsq(health, limits),
        "progressive_174": progresses(paired, limits["core_psnr_gain_db"]),
        "progressive_64": progresses(validation["independent_64"], 0),
        "content_interventions": all(
            row["ci95_low"] > 0 and row["frames"] == 174 for row in interventions.values()
        ),
        "two_sided_chroma": all(chroma_passes(paired[key]) for key in ("4096", HIGH)),
        "absolute_psnr": paired[HIGH]["psnr_global_mse"] >= limits["minimum_k8192_psnr_db"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "advanced_lpips": paired[HIGH]["lpips"] < limits["advanced_k8192_lpips"],
    }


def assess_fix_c(config):
    seeds = {str(seed): assess_seed(directories("fix_c", seed)[0], config) for seed in (1, 2)}
    return {"passed": all(seed["passed"] for seed in seeds.values()), "seeds": seeds}


def decide(phase, path=CONFIG):
    config = fix_settings(phase if phase != "fix_b" else "b0", 1, path)
    if phase == "smoke":
        directory = directories(phase, 1)[0]
        result = assess_smoke(directory, config)
    else:
        directory = RESULTS / phase
        if phase == "fix_a":
            result = assess_fix_a(directories(phase, 1)[0], config)
        elif phase == "fix_b":
            from aip_vtok.evaluation.fix_audit import audit

            integrity = audit()
            write_json(
                directory / "fork_audit.json",
                {**fork_evidence(), "gate0_projector_hash": integrity["passed"]},
            )
            result = assess_fix_b(config)
        else:
            result = assess_fix_c(config)
    result["limits"] = config["fix_limits"]
    write_json(directory / "decision.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=PHASES)
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()
    result = decide(args.phase, args.config)
    print(json.dumps(result))
    if not result["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
