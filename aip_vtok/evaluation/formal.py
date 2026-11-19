"""174-frame paired and independent closed loops plus suffix content interventions."""

import argparse
import csv
import json
from dataclasses import replace

import numpy as np
import torch
from torch.nn import functional as F

from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.evaluation.fix_metrics import fix_chroma
from aip_vtok.evaluation.smoke import BUDGETS, aggregate_budgets, code_health
from aip_vtok.models.codec import create_legacy_codec
from aip_vtok.runtime import file_hash, resolve, write_json
from aip_vtok.training.checkpoint import load_trainable
from aip_vtok.training.data import read_window, source_by_id, sources
from aip_vtok.training.formal import settings
from aip_vtok.training.losses import chroma_diagnostics, latent_mse_metrics
from aip_vtok.training.prefix import prefix_canvas
from cp_adaptok.data import save_rgb
from cp_adaptok.metrics import MetricSuite


def append_until(session, ids, target):
    for offset in range(session.canvas.count, target, 128):
        session.append(ids[:, offset : min(offset + 128, target)], offset)


def render_mutation(codec, encoded, session, ids):
    tokens = replace(encoded.tokens, ids=ids)
    canvas = prefix_canvas(codec.tokenizer, tokens, session.canvas.prediction, 8192, differentiable=False)
    latent = codec.fusion(canvas.latent(), canvas.masks)
    return codec.backbone.decode_aligned(latent, session.context)[0]


def intervention_rows(codec, encoded, session, donor, rgb, baseline_mse, generator):
    rows = []
    for pass_id in (1, 2):
        selected = (encoded.tokens.plan.pass_ids == pass_id).nonzero().flatten()
        permutation = torch.randperm(selected.numel(), generator=generator).to(selected.device)
        for kind in ("shuffle", "cross_video"):
            changed = encoded.tokens.ids.clone()
            changed[:, selected] = (
                changed[:, selected[permutation]] if kind == "shuffle" else donor[:, selected]
            )
            rec = render_mutation(codec, encoded, session, changed)
            mse = F.mse_loss(rgb, rec).item()
            rows.append(
                {
                    "pass": pass_id,
                    "intervention": kind,
                    "mse": mse,
                    "baseline_mse": baseline_mse,
                    "mse_increase": mse - baseline_mse,
                    "relative_mse_increase": mse / baseline_mse - 1,
                }
            )
    return rows


def commit_after_render(session, encoded):
    append_until(session, encoded.tokens.ids, max(session.canvas.count, session.reference_budget))
    return session.commit_reference_once()


def images(directory, video, budget, original, rec):
    label = video.split(":")[-1]
    save_rgb(directory / "images" / f"{label}_original.png", original)
    save_rgb(directory / "images" / f"{label}_original_crop.png", original[..., 400:656, 800:1056])
    save_rgb(directory / "images" / f"{label}_k{budget}.png", rec)
    save_rgb(directory / "images" / f"{label}_k{budget}_crop.png", rec[..., 400:656, 800:1056])


def measure_render(rendered, encoded, suite, rgb, fix):
    if fix:
        rec, raw, fused = rendered
        return {
            **suite(rgb, rec),
            **fix_chroma(rgb, rec),
            **latent_mse_metrics(raw, fused, encoded.native_latent),
        }
    return {**suite(rgb, rendered), **chroma_diagnostics(rgb, rendered)}


def budget_row(config, video, frame_id, paired, budget, mode, values):
    return {
        "seed": config["seed"],
        "mode": mode,
        "video": video,
        "frame": frame_id,
        "frame_type": "I" if paired.context.intra else "P",
        "budget": budget,
        "required_reference_tokens": paired.reference_budget,
        **values,
        "bpp": None,
        "real_bitstream": False,
        "split": "validation",
    }


def evaluate_budget(
    config, codec, suite, video, frame_id, rgb, encoded, paired, independent_state, budget, directory
):
    fix = "model_version" in config
    append_until(paired, encoded.tokens.ids, budget)
    rendered = paired.render_frame(with_latents=fix)
    rec = rendered[0] if fix else rendered
    metrics = measure_render(rendered, encoded, suite, rgb, fix)
    if frame_id == 0 or (fix and directory is not None and frame_id == 63):
        label = video if frame_id == 0 else f"{video}_f063"
        if directory is not None:
            images(directory, label, budget, rgb, rec)
    rows = [budget_row(config, video, frame_id, paired, budget, "paired", metrics)]
    if independent_state is None:
        return rows, None, None, metrics["mse"]
    independent_encoded = codec.encode_frame(rgb, independent_state, frame_id)
    independent = codec.prefix_session(independent_encoded, independent_state, budget)
    independent_render = independent.render_frame(with_latents=fix)
    other = independent_render[0] if fix else independent_render
    error = (rec - other).abs().max().item()
    equal_ids = torch.equal(encoded.tokens.ids, independent_encoded.tokens.ids)
    exact = {
        "video": video,
        "frame": frame_id,
        "budget": budget,
        "max_rgb_error": error,
        "equal_ids": equal_ids,
    }
    if error == 0:
        independent_metrics = metrics
        if fix:
            _, raw, fused = independent_render
            independent_metrics = {
                **metrics,
                **latent_mse_metrics(raw, fused, independent_encoded.native_latent),
            }
    else:
        independent_metrics = measure_render(independent_render, independent_encoded, suite, rgb, fix)
    rows.append(budget_row(config, video, frame_id, paired, budget, "independent", independent_metrics))
    independent_state = commit_after_render(independent, independent_encoded)
    return rows, independent_state, exact, metrics["mse"]


@torch.no_grad()
def evaluate_video(config, codec, suite, video, clip, donor, directory):
    paired_state = ReferenceState()
    independent_states = {budget: ReferenceState() for budget in BUDGETS}
    rows, health, interventions, exact_errors = [], [], [], []
    rng = torch.Generator().manual_seed(7301)
    for frame_id, frame in enumerate(clip):
        rgb = frame[None].cuda().float() / 255
        encoded = codec.encode_frame(rgb, paired_state, frame_id)
        health.append(
            {"video": video, "frame": frame_id, "passes": code_health(encoded, codec.tokenizer.fsq)}
        )
        paired = codec.begin_frame(frame_id, encoded.tokens.tiers, paired_state)
        for budget in BUDGETS:
            measured, independent_states[budget], exact, baseline_mse = evaluate_budget(
                config,
                codec,
                suite,
                video,
                frame_id,
                rgb,
                encoded,
                paired,
                independent_states[budget],
                budget,
                directory,
            )
            rows.extend(measured)
            exact_errors.append(exact)
        interventions.extend(
            {"video": video, "frame": frame_id, **row}
            for row in intervention_rows(codec, encoded, paired, donor, rgb, baseline_mse, rng)
        )
        paired_state = paired.commit_reference_once()
        if (frame_id + 1) % 16 == 0:
            print(f"seed={config['seed']} validation {video} frames={frame_id + 1}", flush=True)
    return rows, health, interventions, exact_errors


def intervention_video_means(rows, pass_id, kind):
    selected = [r for r in rows if r["pass"] == pass_id and r["intervention"] == kind]
    videos = sorted({r["video"] for r in selected})
    means = np.array(
        [np.mean([r["mse_increase"] for r in selected if r["video"] == video]) for video in videos]
    )
    return means, len(selected)


def summarize_interventions(rows):
    rng, result = np.random.default_rng(19), {}
    for pass_id in (1, 2):
        for kind in ("shuffle", "cross_video"):
            means, count = intervention_video_means(rows, pass_id, kind)
            bootstrap = means[rng.integers(len(means), size=(2000, len(means)))].mean(1)
            result[f"pass{pass_id}_{kind}"] = {
                "mean_mse_increase": float(means.mean()),
                "ci95_low": float(np.quantile(bootstrap, 0.025)),
                "ci95_high": float(np.quantile(bootstrap, 0.975)),
                "independent_videos": len(means),
                "frames": count,
            }
    return result


def health_summary(rows):
    result = {}
    for pass_id in range(3):
        selected = [r["passes"][str(pass_id)] for r in rows]
        result[str(pass_id)] = {
            "minimum_unique_ids": min(r["unique_ids"] for r in selected),
            "minimum_id_entropy_bits": min(r["id_entropy_bits"] for r in selected),
            "minimum_dimension_entropy_bits": min(min(r["dimension_entropy_bits"]) for r in selected),
        }
    return result


def evaluation_assets(config):
    directory = resolve(config["output"]) / f"gate1_seed{config['seed']}"
    checkpoint = resolve(config["checkpoints"]) / f"gate1_seed{config['seed']}/latest.pt"
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if artifact["step"] != config["stage1"]["steps"] or artifact["config"] != config:
        raise ValueError("formal evaluation requires the declared completed checkpoint")
    codec = create_legacy_codec(config).eval()
    load_trainable(codec, artifact["model"])
    suite = metric_suite()
    selected, clips, hashes = validation_inputs(config)
    return directory, checkpoint, codec, suite, selected, clips, hashes


def metric_suite():
    suite = MetricSuite("cuda:0")
    if suite.errors:
        raise RuntimeError(f"required metrics unavailable: {suite.errors}")
    return suite


def validation_inputs(config):
    available, hashes = sources(config, "validation")
    protocol = config["stage1"]["validation"]
    selected = [source_by_id(available, video) for video in protocol["videos"]]
    clips = [
        read_window(row, 0, min(row["original_frames"], protocol["max_frames_per_video"])) for row in selected
    ]
    if sum(len(clip) for clip in clips) != protocol["total_frames"]:
        raise ValueError("validation protocol is not exactly 174 original frames")
    return selected, clips, hashes


def evaluate(config):
    directory, checkpoint, codec, suite, selected, clips, hashes = evaluation_assets(config)
    donors = []
    with torch.no_grad():
        for clip in clips:
            donors.append(codec.encode_frame(clip[:1].cuda().float() / 255, ReferenceState(), 0).tokens.ids)
    rows, health, interventions, exact = [], [], [], []
    for index, (row, clip) in enumerate(zip(selected, clips)):
        measured = evaluate_video(
            config, codec, suite, row["content_id"], clip, donors[(index + 1) % len(donors)], directory
        )
        rows.extend(measured[0])
        health.extend(measured[1])
        interventions.extend(measured[2])
        exact.extend(measured[3])
    return store_evaluation(config, directory, checkpoint, hashes, rows, health, interventions, exact)


def store_evaluation(config, directory, checkpoint, hashes, rows, health, interventions, exact):
    summary = {
        mode: aggregate_budgets([r for r in rows if r["mode"] == mode]) for mode in ("paired", "independent")
    }
    artifact = {
        "seed": config["seed"],
        "training_steps": config["stage1"]["steps"],
        "summary": summary,
        "health_summary": health_summary(health),
        "intervention_summary": summarize_interventions(interventions),
        "maximum_independent_rgb_error": max(r["max_rgb_error"] for r in exact),
        "independent_ids_all_equal": all(r["equal_ids"] for r in exact),
        "manifest_hashes": hashes,
        "checkpoint_sha256": file_hash(checkpoint),
        "real_bitstream": False,
        "bpp": None,
        "precision": "FP32",
        "metric_reuse": "independent metrics reused only after exact full-RGB equality is checked",
        "reference_policy": "I2048/P510; I renders below 2048 still receive reference prefix before commit",
    }
    write_json(directory / "health.json", health)
    write_json(directory / "interventions.json", interventions)
    write_json(directory / "independent_exactness.json", exact)
    write_frame_metrics(directory, rows)
    write_json(directory / "evaluation.json", artifact)
    return artifact


def write_frame_metrics(directory, rows):
    with (directory / "frame_metrics.csv.tmp").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (directory / "frame_metrics.csv.tmp").replace(directory / "frame_metrics.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/aip_vtok/gate1.yaml")
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    result = evaluate(settings(args.config, args.seed))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
