"""Paired budget evaluation under one fixed-prefix history, no test-set access."""

import csv

import torch

from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.evaluation.gates import distortion_summary
from aip_vtok.runtime import resolve, write_json
from aip_vtok.training.losses import chroma_retention
from cp_adaptok.data import save_rgb

BUDGETS = (510, 1024, 2048, 4096, 8192)


def entropy(values, size):
    counts = torch.bincount(values.flatten().cpu(), minlength=size).double()
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * probabilities.log2()).sum())


def code_health(encoded, fsq):
    result = {}
    for pass_id in range(3):
        selected = encoded.tokens.plan.pass_ids == pass_id
        ids = encoded.tokens.ids[:, selected]
        digits = fsq.ids_to_digits(ids)
        result[str(pass_id)] = {
            "tokens": ids.numel(),
            "unique_ids": ids.unique().numel(),
            "id_entropy_bits": entropy(ids, 64000),
            "dimension_entropy_bits": [entropy(digits[..., i], int(fsq.levels[i])) for i in range(6)],
        }
    return result


def aggregate_budgets(rows):
    result = {}
    for budget in BUDGETS:
        selected = [r for r in rows if r["budget"] == budget]
        result[str(budget)] = {
            **distortion_summary(selected),
            **{
                name: sum(r[name] for r in selected) / len(selected)
                for name in (
                    "lpips",
                    "dists",
                    "ms-ssim",
                    "chroma_retention",
                    "chroma_mse",
                    "chroma_mean_abs_bias",
                    "source_chroma_variance",
                )
                if name in selected[0]
            },
        }
    return result


@torch.no_grad()
def evaluate_smoke(config, codec, clip, suite, step):
    codec.eval()
    rows, health, state = [], [], ReferenceState()
    directory = resolve(config["output"]) / "gate1_smoke"
    for frame_id, frame in enumerate(clip):
        rgb = frame[None].cuda().float() / 255
        encoded = codec.encode_frame(rgb, state, frame_id)
        health.append({"frame": frame_id, "passes": code_health(encoded, codec.tokenizer.fsq)})
        for budget in BUDGETS:
            session = codec.prefix_session(encoded, state, budget)
            rec = session.render_frame()
            rows.append(
                {
                    "step": step,
                    "frame": frame_id,
                    "budget": budget,
                    **suite(rgb, rec),
                    "chroma_retention": chroma_retention(rgb, rec),
                    "bpp": None,
                    "split": "train",
                    "mode": "paired_overfit_smoke",
                    "real_bitstream": False,
                }
            )
            if step in (0, config["stage1"]["steps"]) and frame_id == 0:
                save_rgb(directory / "images" / f"step{step:03d}_k{budget}.png", rec)
                save_rgb(
                    directory / "images" / f"step{step:03d}_k{budget}_crop.png", rec[..., 400:656, 800:1056]
                )
        if step == 0 and frame_id == 0:
            save_rgb(directory / "images/original.png", rgb)
            save_rgb(directory / "images/original_crop.png", rgb[..., 400:656, 800:1056])
        reference_budget = codec.reference_i if frame_id % codec.backbone.gop == 0 else codec.reference_p
        reference_session = codec.prefix_session(encoded, state, reference_budget)
        state = reference_session.commit_reference_once()
    result = {
        "step": step,
        "summary": aggregate_budgets(rows),
        "rows": rows,
        "health": health,
        "data_policy": "training video only; not the 174-frame validation gate",
    }
    write_json(directory / f"metrics_{step:06d}.json", result)
    codec.train()
    return result


def write_curve(directory):
    import json

    rows = []
    for path in sorted(directory.glob("metrics_*.json")):
        rows.extend(json.loads(path.read_text())["rows"])
    with (directory / "rd_curve.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
