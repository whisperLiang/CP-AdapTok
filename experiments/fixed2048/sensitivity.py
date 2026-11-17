"""Separate nonstandard token-content diagnostics with explicit history ownership."""

import argparse

import torch

from cp_adaptok.runtime import resolve, write_json
from experiments.fixed2048.checkpoint import load_reconstructor
from experiments.fixed2048.cli import seeded_config
from experiments.fixed2048.data import read_clip, records
from experiments.fixed2048.diagnostics import content_sensitivity
from experiments.fixed2048.evaluate import FLAGS, force_intra


@torch.no_grad()
def diagnose_sequence(model, source, clip, path, reference):
    state, previous, diagnosed, rows = model.empty_state(), {}, set(), []
    for index, frame in enumerate(clip):
        rgb = frame[None].to(model.device)
        encoded = model.encode(rgb, state, index, force_intra=force_intra(reference, index))
        normal = model.reconstruct(encoded, path)
        frame_type = encoded.context.frame_type
        if frame_type in previous and frame_type not in diagnosed:
            peer_index, peer_features = previous[frame_type]
            values = content_sensitivity(model, encoded, peer_features, path, target=rgb)
            if encoded.context.committed:
                raise RuntimeError("anomalous content must never advance the state")
            rows.append(
                {
                    "video": source["content_id"],
                    "frame": index,
                    "peer_frame": peer_index,
                    "frame_type": frame_type,
                    "history_path": path,
                    "diagnostic_path": path,
                    "reference": reference,
                    "fixed_positions_and_context": True,
                    "diagnostic_only": True,
                    "results": values,
                    **FLAGS,
                }
            )
            diagnosed.add(frame_type)
        previous[frame_type] = index, encoded.features.detach()
        state = model.advance(encoded, normal, reference)
    return rows


def run(config, checkpoint, output):
    output = resolve(output)
    if output.exists():
        raise FileExistsError("content-diagnostic output exists")
    model, identity = load_reconstructor(checkpoint, config["device"])
    selected, hashes = records(config, "validation", config["validation"]["videos"])
    measured = []
    for source in selected:
        clip = read_clip(source, 4)
        for reference in ("i_only", "full2048", "original128"):
            for path in ("R1", "R2"):
                measured.extend(diagnose_sequence(model, source, clip, path, reference))
    write_json(
        output,
        {
            "checkpoint": identity,
            "manifest_hashes": hashes,
            "rows": measured,
            "normal_state_only": True,
            **FLAGS,
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fixed2048.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run(seeded_config(args.config), args.checkpoint, args.output)


if __name__ == "__main__":
    main()
