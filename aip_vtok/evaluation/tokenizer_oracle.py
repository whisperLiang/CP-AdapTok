"""Same-context least-squares diagnostic of the learned lift at each pass."""

import argparse

import torch

from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.evaluation.fix import load_evaluation_codec, rgb_frame
from aip_vtok.runtime import file_hash, write_json
from aip_vtok.training.data import read_window, source_by_id, sources
from aip_vtok.training.determinism import configure_determinism
from aip_vtok.training.fix import CONFIG, directories, fix_settings, require_release


def unpack_patches(latent, size):
    batch, channels, height, width = latent.shape
    grid = latent.reshape(batch, channels, height // size, size, width // size, size)
    return grid.permute(0, 2, 4, 1, 3, 5).reshape(batch, -1, channels * size**2)


def continuous_codes(lift, target_patches, inverse):
    return (target_patches - lift.bias) @ inverse.T


def pass_rows(codec, encoded, inverses):
    tokenizer, tokens = codec.tokenizer, encoded.tokens
    target, raw = encoded.native_latent, encoded.prediction
    rows = []
    for p, size in enumerate((4, 2, 1)):
        chosen = tokens.plan.pass_ids == p
        positions = tokens.plan.positions[chosen]
        shape = tokens.plan.latent_shape
        error = target - raw
        best_codes = continuous_codes(
            tokenizer.lifts[p], unpack_patches(error, size)[:, positions], inverses[p]
        )
        best_pass = tokenizer.synthesize_pass(best_codes, positions, p, shape)
        actual_pass = tokenizer.synthesize_pass(tokens.codes[:, chosen], positions, p, shape)
        raw = raw + actual_pass
        rows.append(
            {
                "pass": p,
                "active_fraction": positions.numel() / (shape[0] * shape[1] / size**2),
                "raw_mse_before": error.square().mean().item(),
                "raw_mse_after_fsq": (target - raw).square().mean().item(),
                "raw_mse_after_oracle": (error - best_pass).square().mean().item(),
            }
        )
    return rows


@torch.no_grad()
def diagnose(phase, seed, config_path):
    release = require_release(phase)
    config = fix_settings(phase, seed, config_path, joint_projector=release.get("winner") == "b1")
    configure_determinism()
    directory, weights = directories(phase, seed)
    checkpoint = weights / "latest.pt"
    codec = load_evaluation_codec(config, directory, checkpoint, config["stage1"]["steps"])
    inverses = [
        torch.linalg.pinv(lift.weight.cpu().double()).to(lift.weight) for lift in codec.tokenizer.lifts
    ]
    available, _ = sources(config, "validation")
    rows = []
    for video in config["stage1"]["validation"]["videos"]:
        state = ReferenceState()
        clip = read_window(source_by_id(available, video), 0, 2)
        for frame_id, frame in enumerate(clip):
            rgb = rgb_frame(frame)
            encoded = codec.encode_frame(rgb, state, frame_id)
            rows.extend(
                {"video": video, "frame": frame_id, **row} for row in pass_rows(codec, encoded, inverses)
            )
            reference = codec.reference_i if encoded.context.intra else codec.reference_p
            state = codec.prefix_session(encoded, state, reference).commit_reference_once()
    write_json(
        directory / "tokenizer_oracle.json",
        {
            "checkpoint_sha256": file_hash(checkpoint),
            "rows": rows,
            "protocol": "one pass replaced with its unconstrained least-squares 6D codes; earlier passes and committed history are measured FSQ",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("fix_a", "fix_c"))
    parser.add_argument("--seed", type=int, choices=(1, 2), default=1)
    parser.add_argument("--config", default=CONFIG)
    args = parser.parse_args()
    diagnose(args.phase, args.seed, args.config)


if __name__ == "__main__":
    main()
