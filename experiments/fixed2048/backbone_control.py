"""Separate continuous I-latent scale control grounded in official DCVC code.

The task's R0 raw-y definition is retained in codec.py. Official image prior
multiplies by q_scale_y_enc before reconstruction and q_scale_y_dec afterwards.
Bypassing rounding must retain both factors. This control does not call priors,
entropy models, native compression, or any bitstream method.
"""

import argparse

import torch
from torch.nn import functional as F

from cp_adaptok.metrics import MetricSuite
from cp_adaptok.runtime import resolve, write_json
from experiments.fixed2048.checkpoint import load_reconstructor
from experiments.fixed2048.cli import seeded_config
from experiments.fixed2048.data import read_clip, records
from experiments.fixed2048.diagnostics import distortion
from experiments.fixed2048.evaluate import FLAGS, save_image


def continuous_intra_latent(adapter, latent, qp):
    return latent * adapter.quant("q_scale_y_enc", qp) * adapter.quant("q_scale_y_dec", qp)


@torch.no_grad()
def control_loop(model, source, clip, reference, spatial, suite, directory):
    state, rows = model.empty_state(), []
    path = "R0-spatial-scale-control" if spatial else "R0-scale-control"
    for index, frame in enumerate(clip):
        rgb = frame[None].to(model.device)
        encoded = model.encode(rgb, state, index, force_intra=reference == "i_only")
        latent = encoded.latent
        if encoded.context.frame_type == "I":
            latent = continuous_intra_latent(model.intra, latent, model.config.qp)
        if spatial:
            low = F.adaptive_avg_pool2d(latent, (17, 30))
            latent = F.interpolate(low, size=latent.shape[-2:], mode="bilinear", align_corners=False)
        rec, feature = model.decode_latent(latent, encoded.context)
        rows.append(
            {
                "video": source["content_id"],
                "frame": index,
                "frame_type": encoded.context.frame_type,
                "reference": reference,
                "path": path,
                "pixels": 1080 * 1920,
                **suite(rgb, rec),
                **distortion(encoded.latent, latent),
                **FLAGS,
            }
        )
        if index == 0:
            save_image(
                directory / "images" / f"{source['content_id'].split(':')[-1]}_{reference}_{path}.png", rec
            )
        if reference == "full2048":
            state = encoded.context.commit(feature, f"scale-control:{index}")
    return rows


def run_control(config, output, frames=8):
    directory = resolve(output)
    if directory.exists():
        raise FileExistsError("scale-control output exists")
    model, identity = load_reconstructor(config["source_checkpoint"], config["device"])
    suite = MetricSuite(model.device)
    if suite.errors:
        raise RuntimeError(f"required metrics unavailable: {suite.errors}")
    selected, hashes = records(config, "validation", config["baseline"]["videos"])
    rows = []
    for source in selected:
        clip = read_clip(source, frames)
        for reference in ("i_only", "full2048"):
            for spatial in (False, True):
                rows.extend(control_loop(model, source, clip, reference, spatial, suite, directory))
    write_json(directory / "frame_metrics.json", rows)
    from experiments.fixed2048.evaluate import write_tables

    write_tables(directory, rows)
    write_json(
        directory / "run.json",
        {
            "checkpoint": identity,
            "manifest_hashes": hashes,
            "frames": len(rows),
            "official_source": "third_party/DCVC/src/models/common_model.py:forward_prior_4x",
            "note": "retains both I-latent scale factors, bypasses all rounding; separate from raw-y R0",
            **FLAGS,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fixed2048.yaml")
    parser.add_argument("--output", default="results/fixed2048/baseline_scale_control")
    parser.add_argument("--frames", type=int, default=8)
    args = parser.parse_args()
    config = seeded_config(args.config)
    run_control(config, args.output, args.frames)


if __name__ == "__main__":
    main()
