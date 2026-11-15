"""Reproduce checkpoint collapse on real frames without changing model weights."""

import argparse

import torch
from torch.nn import functional as F

from cp_adaptok.data import read_video
from cp_adaptok.diagnostics import vq_diagnostics
from cp_adaptok.runtime import build_model, metadata, read_config, seed_everything, write_json
from cp_adaptok.streaming.sender import ids_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="results/optimization_20261115/baseline_diagnostics.json")
    args = parser.parse_args()
    config = read_config("configs/pretrained_s1.yaml")
    seed_everything(config["seed"], config["device"], 0.6)
    model, _, _ = build_model(
        config, config["device"], checkpoint_override="checkpoints/pretrained_s1/step100.pt"
    )
    model.eval()
    clip = read_video(dict(config["data"]["train"][0], frames=4), 256, torch.Generator().manual_seed(1))
    state, rows = model.empty_state(), []
    with torch.no_grad():
        for t, frame in enumerate(clip):
            rgb = frame[None].to(model.device)
            encoded = model.encode_full(rgb, state)
            row = {
                "frame": t,
                "type": encoded.context.frame_type,
                **vq_diagnostics(model.vq, encoded.features, encoded.vq),
            }
            row["prefixes"], previous = [], None
            for k in config["training"]["budgets"]:
                latent = model.decoder(
                    encoded.vq.embeddings[:, :k], encoded.context.latent_shape, encoded.context.frame_type
                )
                rec, _ = model.decode_latent(latent, encoded.context)
                row["prefixes"].append(
                    {
                        "tokens": k,
                        "mse": F.mse_loss(rec, rgb).item(),
                        "latent_delta_mse": F.mse_loss(latent, previous).item()
                        if previous is not None
                        else None,
                    }
                )
                previous = latent
            _, feature = model.decode_ids(encoded.prefix(model.config.k_ref), encoded.context)
            state = encoded.context.commit(feature, ids_digest(encoded.prefix(model.config.k_ref)))
            rows.append(row)
    write_json(args.output, {**metadata(config, "baseline_diagnostic"), "records": rows})
    print(
        [
            {k: r[k] for k in ["frame", "type", "pre_vq", "nearest_distance_mean"]}
            | {"deterministic_codes": r["deterministic"]["used_codes"]}
            for r in rows
        ]
    )


if __name__ == "__main__":
    main()
