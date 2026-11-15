"""Measure content, attention, counterfactuals and true per-budget gradients."""

import argparse
import csv

import torch

from cp_adaptok.data import read_video
from cp_adaptok.diagnostics_prefix import diagnose_clip
from cp_adaptok.runtime import build_model, metadata, read_config, resolve, seed_everything, write_json


def save_diagnostics(result, output):
    output = resolve(output)
    output.mkdir(parents=True, exist_ok=True)
    for key in ("per_group_statistics", "attention_mass", "gradient_by_group", "token_interventions"):
        write_json(output / f"{key}.json", result[key])
    rows = result["token_interventions"]
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (output / "reconstruction_sensitivity.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    plot_diagnostics(result, output)


def plot_diagnostics(result, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    last = result["attention_mass"][-1]["prefixes"][-1]
    fig, ax = plt.subplots(figsize=(9, 4))
    im = ax.imshow(last["per_grid_group_attention_mass"], aspect="auto")
    ax.set(xlabel="Packet group", ylabel="Grid query", title="Attention mass (not independent information)")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output / "attention_visualization.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 4))
    for row in result["gradient_by_group"][-1]["budgets"]:
        ax.plot([g["rms"] for g in row["decoder_embedding"]], label=str(row["tokens"]), marker=".")
    ax.set(xlabel="Packet group", ylabel="Embedding gradient RMS", yscale="symlog",
           title="Actual reconstruction MSE gradient")
    ax.legend(title="Prefix tokens")
    fig.tight_layout()
    fig.savefig(output / "gradient_distribution.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/prefix_diagnostics.yaml")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="results/prefix_utilization/diagnostics")
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--crop", type=int)
    args = parser.parse_args()
    config = read_config(args.config)
    seed_everything(config["seed"], config["device"], config["gpu_memory_fraction"])
    model, _, _ = build_model(config, config["device"], checkpoint_override=args.checkpoint)
    spec = dict(config["data"]["test"][0], frames=args.frames)
    clip = read_video(spec, args.crop, torch.Generator().manual_seed(19))
    result = diagnose_clip(model, clip, config["evaluation"]["budgets"])
    save_diagnostics(result, args.output)
    write_json(resolve(args.output) / "manifest.json", dict(
        metadata=metadata(config), checkpoint=model.artifact_provenance, source=spec,
        input_shape=list(clip.shape), metric_errors=result["metric_errors"], semantics=result["semantics"]))


if __name__ == "__main__":
    main()
