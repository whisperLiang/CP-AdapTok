"""Independent deterministic quality validation and train-only real-stream RD."""

import copy
import math

import torch

from cp_adaptok.data import read_video, sample_training_spec
from cp_adaptok.evaluation import (
    _quality_summary,
    budget_summary,
    evaluate_sequences,
    frame_type_summaries,
    prefix_gain_summary,
)
from cp_adaptok.large_scale_data import video_specs
from cp_adaptok.metrics import MetricSuite
from cp_adaptok.runtime import load_checkpoint, read_config, resolve, write_json
from cp_adaptok.streaming.sender import ids_digest


def selection_score(report, weights=None):
    weights = weights or read_config("configs/large_scale/validation.yaml")["selection"]
    rows = report["budget_summary"]
    distortion = sum(row["mse"] for row in rows) / len(rows)
    endpoints = rows[0]["mse"] + rows[-1]["mse"]
    score = distortion + weights["endpoint_weight"] * endpoints
    score += weights["regression_weight"] * report.get("quality_drop_ratio", 0)
    for metric in ("lpips", "dists", "bpp"):
        values = [row[metric] for row in rows if row.get(metric) is not None]
        score += weights[f"{metric}_weight"] * sum(values) / max(1, len(values))
    if not math.isfinite(score):
        raise ValueError("nonfinite validation score")
    return score


@torch.no_grad()
def calibrate_entropy(model, config):
    """Deterministic discrete tokens from train only, with CP's own causal base history."""
    settings = config["large_scale"]["entropy_calibration"]
    model.eval()
    model.refresh_identity()
    counts = torch.full_like(model.entropy_model.logits, settings["pseudocount"], dtype=torch.float64)
    generator = torch.Generator().manual_seed(config["seed"] + 9011)
    sources = config["data"]["train"]
    indices = torch.randperm(len(sources), generator=generator)[:settings["clips"]].tolist()
    observations, windows = 0, []
    for index in indices:
        spec = sample_training_spec(sources[index], settings["frames"], generator)
        clip = read_video(spec, settings["crop"], generator)
        state = model.empty_state()
        windows.append({"path": spec.path, "start": spec.start, "frames": spec.frames})
        for frame in clip:
            encoded = model.encode_full(frame[None].to(model.device), state)
            grouped = encoded.vq.ids.view(-1, model.config.packet_tokens)
            counts.scatter_add_(1, grouped, torch.ones_like(grouped, dtype=counts.dtype))
            _, feature = model.decode_ids(encoded.prefix(model.config.k_ref), encoded.context)
            state = encoded.context.commit(feature, ids_digest(encoded.prefix(model.config.k_ref))).detach()
            observations += grouped.numel()
    model.entropy_model.logits.copy_(counts.log().to(model.entropy_model.logits))
    model.refresh_identity()
    return {"method": "train-only deterministic-token multinomial MLE with additive smoothing",
            "observed_tokens": observations, "windows": windows, "validation_or_test_tokens_used": False,
            "entropy_id": model.entropy_model.export().entropy_id, "model_id": model.model_id,
            "manifest_hashes": config.get("experiment", {}).get("manifest_hashes")}


def validation_specs(splits, tier, settings):
    rows = splits["test" if tier == "final" else "validation"]
    dimensions = settings.get("crop")
    if tier != "quick" and dimensions:
        # Report native1080 RD without averaging 4K and 1080p denominators together.
        rows = [row for row in rows if row["resolution"] == list(reversed(dimensions))]
    specs = video_specs(rows, crop=dimensions, frames=settings["frames"])
    count = settings["sequences"]
    return specs[:count] if count is not None else specs


@torch.no_grad()
def quick_validation(model, config, specs, settings):
    """PSNR/MSE and discrete-code diagnostics; never present this as measured RD."""
    metrics = MetricSuite(model.device, settings["metrics"])
    records, perplexity, usage = [], [], []
    generator = torch.Generator().manual_seed(config["seed"] + 8009)
    for spec in specs:
        state = model.empty_state()
        clip = read_video(spec, settings.get("crop"), generator)
        for t, frame in enumerate(clip):
            rgb = frame[None].to(model.device)
            encoded = model.encode_full(rgb, state)
            rows = []
            base_feature = None
            for k in config["evaluation"]["budgets"]:
                reconstructed, feature = model.decode_ids(encoded.prefix(k), encoded.context)
                rows.append({"tokens": k, **metrics(rgb, reconstructed)})
                if k == model.config.k_ref:
                    base_feature = feature
            state = encoded.context.commit(base_feature, ids_digest(encoded.prefix(model.config.k_ref))).detach()
            records.append({"sequence": spec["content_id"], "source": spec, "frame_id": t,
                            "frame_type": encoded.context.frame_type, "prefixes": rows})
            perplexity.append(encoded.vq.perplexity.item())
            usage.append(encoded.vq.usage.item())
    if not records:
        raise ValueError("no independent validation sequences at the configured native crop")
    budgets = config["evaluation"]["budgets"]
    return {"records": records, "budget_summary": budget_summary(records, budgets),
            "budget_summary_by_frame_type": frame_type_summaries(records, budgets),
            "prefix_gains": prefix_gain_summary(records), **_quality_summary(records, False),
            "deterministic_perplexity": sum(perplexity) / len(perplexity),
            "deterministic_code_usage": sum(usage) / len(usage), "metric_errors": metrics.errors,
            "real_entropy_bitstream": False, "rate_interpretation": "quality-only; no bitrate claim"}


def validate_checkpoint(model, config, splits, tier, output, checkpoint=None, settings_override=None):
    settings = read_config("configs/large_scale/validation.yaml")[tier]
    if settings_override:
        settings.update(settings_override)
    specs = validation_specs(splits, tier, settings)
    output = resolve(output)
    output.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    # Perceptual constructors must not disturb subsequent training, even on old checkpoints.
    devices = [model.device.index or 0] if model.device.type == "cuda" else []
    logits = model.entropy_model.logits.detach().clone()
    try:
        with torch.random.fork_rng(devices=devices):
            model.eval()
            model.refresh_identity()
            if tier == "quick":
                report = quick_validation(model, config, specs, settings)
            else:
                report = real_rd_validation(model, config, specs, settings, output, checkpoint,
                                            "test" if tier == "final" else "validation")
            report.update(tier=tier, split="test" if tier == "final" else "validation",
                          quantization="deterministic_nearest", checkpoint=str(checkpoint),
                          model_selection_allowed=tier != "final")
            write_json(output / "validation.json", report)
            return report
    finally:
        with torch.no_grad():
            model.entropy_model.logits.copy_(logits)
        model.refresh_identity()
        model.train(was_training)


def real_rd_validation(model, config, specs, settings, output, checkpoint, split):
    calibration = calibrate_entropy(model, config)
    write_json(output / "entropy_calibration.json", calibration)
    if checkpoint:
        artifact = load_checkpoint(checkpoint)
        artifact.update(model=model.state_dict(), entropy_calibration=calibration, inference_only=True)
        artifact.pop("optimizer", None)
        artifact.pop("scheduler", None)
        torch.save(artifact, output / "calibrated.pt")
    configured = copy.deepcopy(config)
    configured["data"][split] = specs
    # Native full frames; resolution is never silently reduced for RD reporting.
    report = evaluate_sequences(model, configured, split=split, metric_names=settings["metrics"], status="trained",
                                output=output, compare_cache=True)
    report.update(entropy_calibration=calibration, entropy_id=calibration["entropy_id"],
                  trained_rd_validated=True, scorer_status="requires post-training Stage3 retraining",
                  interpretation="held-out real rANS quality/RD; no convergence or multi-seed claim")
    return report
