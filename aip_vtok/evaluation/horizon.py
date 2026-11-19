"""Bounded causal counterfactuals for a failed Gate 1; no optimizer or new gate."""

import torch
from torch import nn
from torch.nn import functional as F

from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.evaluation.diagnose import aligned_target, capture_encoders
from aip_vtok.evaluation.gates import distortion_summary
from aip_vtok.evaluation.smoke import entropy
from aip_vtok.models.codec import create_legacy_codec
from aip_vtok.runtime import resolve, write_json
from aip_vtok.training.checkpoint import load_trainable
from aip_vtok.training.data import read_window, source_by_id, sources
from aip_vtok.training.formal import settings


class ZeroPrediction(nn.Module):
    def forward(self, previous, intra, shape, device):
        return torch.zeros(1, 128, *shape, device=device)


def rms(tensor):
    return None if tensor is None else tensor.double().square().mean().sqrt().item()


def scalar_health(codec, encoded):
    selected = encoded.tokens.plan.pass_ids == 0
    digits = codec.tokenizer.fsq.ids_to_digits(encoded.tokens.ids[:, selected])
    boundary = ((digits == 0) | (digits == codec.tokenizer.fsq.levels - 1)).float().mean().item()
    entropies = [entropy(digits[..., i], int(codec.tokenizer.fsq.levels[i])) for i in range(6)]
    return {"anchor_boundary_fraction": boundary, "anchor_min_dimension_entropy": min(entropies)}


def configured_codec(config, mode):
    codec = create_legacy_codec(config).eval()
    path = resolve(config["checkpoints"]) / "gate1_seed1/latest.pt"
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    load_trainable(codec, artifact["model"])
    if mode == "zero_prediction_gop32":
        codec.predictor = ZeroPrediction()
    if mode == "reset_gop4":
        codec.backbone.gop = 4
    return codec


@torch.no_grad()
def run_mode(config, clip, mode):
    codec = configured_codec(config, mode)
    captured, state, rows = {}, ReferenceState(), []
    hooks = capture_encoders(codec, captured)
    try:
        for frame_id, frame in enumerate(clip):
            rgb = frame[None].cuda().float() / 255
            encoded = codec.encode_frame(rgb, state, frame_id)
            session = codec.prefix_session(encoded, state, 8192)
            rec = session.render_frame()
            target = aligned_target(codec, captured, session.context.intra)
            direct_mse = None
            if frame_id in (0, 3, 7, 15, 31, 32, 47, 63):
                direct = codec.backbone.decode_aligned(target, session.context)[0]
                direct_mse = F.mse_loss(rgb, direct).item()
            next_state = session.commit_reference_once()
            rows.append(
                {
                    "frame": frame_id,
                    "frame_type": "I" if session.context.intra else "P",
                    "mse": F.mse_loss(rgb, rec).item(),
                    "direct_mse": direct_mse,
                    "reference_latent_rms": rms(next_state.latent),
                    "reference_feature_rms": rms(next_state.feature),
                    "memory_rms": rms(next_state.memory),
                    "native_latent_rms": rms(target),
                    "prediction_rms": rms(session.canvas.prediction),
                    "true_residual1_rms": rms(encoded.tokens.residual1),
                    **scalar_health(codec, encoded),
                }
            )
            state = next_state
    finally:
        for hook in hooks:
            hook.remove()
    return {
        "mode": mode,
        "rows": rows,
        "summary": distortion_summary(rows),
        "early_1_3": distortion_summary([r for r in rows if 1 <= r["frame"] <= 3]),
        "late_48_63": distortion_summary([r for r in rows if 48 <= r["frame"] <= 63]),
        "encoder_calls": codec.encode_calls,
        "gop": codec.backbone.gop,
    }


def main():
    config = settings("configs/aip_vtok/gate1.yaml", 1)
    rows, _ = sources(config, "validation")
    clip = read_window(source_by_id(rows, "DAVIS2017:car-roundabout"), 0, 64)
    results = []
    for mode in ("original_gop32", "zero_prediction_gop32", "reset_gop4"):
        result = run_mode(config, clip, mode)
        results.append(result)
        print(mode, result["summary"], result["late_48_63"], flush=True)
    write_json(
        "results/aip_vtok/gate1_horizon_diagnostics.json",
        {
            "video": "DAVIS2017:car-roundabout",
            "seed": 1,
            "budget": 8192,
            "results": results,
            "scope": "post-failure causal diagnosis; no optimizer; GOP4 is not the GOP32 acceptance result",
            "reference_policy": "I2048/P510 in every mode; enhancements never commit",
        },
    )


if __name__ == "__main__":
    main()
