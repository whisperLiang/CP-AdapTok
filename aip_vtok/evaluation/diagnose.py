"""Matched-history representation/gradient diagnosis; never updates a model."""

import argparse
import json

import torch
from torch.nn import functional as F

from aip_vtok.backbones.dcvc_aligned import continuous_intra_latent
from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.evaluation.gates import distortion_summary
from aip_vtok.evaluation.smoke import BUDGETS
from aip_vtok.models.codec import create_legacy_codec
from aip_vtok.runtime import file_hash, resolve, write_json
from aip_vtok.training.checkpoint import load_trainable
from aip_vtok.training.data import read_window, source_by_id, sources
from aip_vtok.training.formal import settings
from aip_vtok.training.losses import ReconstructionLoss, crop_plan
from aip_vtok.training.prefix import prefix_canvas, receiver_context


def capture_encoders(codec, captured):
    def capture(kind):
        def hook(module, args, output):
            captured[kind] = output.detach()

        return hook

    return [
        codec.backbone.intra.model.enc.register_forward_hook(capture("I")),
        codec.backbone.ld.model.encoder.register_forward_hook(capture("P")),
    ]


def aligned_target(codec, captured, intra):
    if not intra:
        return codec.backbone.alignment.align(captured["P"], False)
    adapter, qp = codec.backbone.intra, codec.backbone.qp
    latent = continuous_intra_latent(
        captured["I"], adapter.quant("q_scale_y_enc", qp), adapter.quant("q_scale_y_dec", qp)
    )
    return codec.backbone.alignment.align(latent, True)


def rms(tensor):
    return tensor.float().square().mean().sqrt().item()


def probe_gradient(codec, objective, rgb, latent, context):
    leaf = latent.detach().requires_grad_(True)
    rec = codec.backbone.decode_aligned(leaf, context, checkpointed=True)[0]
    crops = crop_plan(1080, 1920, torch.Generator().manual_seed(719))
    _, terms = objective(rgb, rec, crops)
    return {
        name: {
            "weighted_value": float(objective.weights[name] * value.detach()),
            "gradient_norm_at_aligned_latent": float(
                torch.autograd.grad(objective.weights[name] * value, leaf, retain_graph=True)[0].norm()
            ),
        }
        for name, value in terms.items()
    }


def diagnostic_frame(codec, encoded, state, captured, rgb, objective, frame_id, video):
    with torch.no_grad():
        context, prediction = receiver_context(codec, state, frame_id, encoded.tokens)
        target = aligned_target(codec, captured, context.intra)
        direct = codec.backbone.decode_aligned(target, context)[0]
        direct_mse = F.mse_loss(rgb, direct).item()
    rows, gradients = [], {}
    for budget in BUDGETS:
        with torch.no_grad():
            canvas = prefix_canvas(codec.tokenizer, encoded.tokens, prediction, budget, differentiable=False)
            raw = canvas.latent()
            fused = codec.fusion(raw, canvas.masks)
            rec = codec.backbone.decode_aligned(fused, context)[0]
            rows.append(
                {
                    "video": video,
                    "frame": frame_id,
                    "frame_type": "I" if context.intra else "P",
                    "budget": budget,
                    "mse": F.mse_loss(rgb, rec).item(),
                    "direct_mse": direct_mse,
                    "raw_latent_mse": F.mse_loss(raw, target).item(),
                    "fused_latent_mse": F.mse_loss(fused, target).item(),
                    "native_latent_rms": rms(target),
                    "prediction_rms": rms(prediction),
                    "pass_rms": [rms(buffer) for buffer in canvas.buffers],
                }
            )
        if budget == 4096:
            gradients = probe_gradient(codec, objective, rgb, fused, context)
    with torch.no_grad():
        reference = codec.prefix_session(
            encoded, state, codec.reference_i if context.intra else codec.reference_p
        )
        state = reference.commit_reference_once()
    return rows, gradients, state


def summarize(rows):
    result = {}
    for budget in BUDGETS:
        selected = [r for r in rows if r["budget"] == budget]
        result[str(budget)] = {
            **distortion_summary(selected),
            "matched_direct": distortion_summary([{"mse": r["direct_mse"]} for r in selected]),
            "raw_latent_mse": sum(r["raw_latent_mse"] for r in selected) / len(selected),
            "fused_latent_mse": sum(r["fused_latent_mse"] for r in selected) / len(selected),
        }
    return result


def diagnose(config):
    directory = resolve(config["output"]) / f"gate1_seed{config['seed']}"
    checkpoint = resolve(config["checkpoints"]) / f"gate1_seed{config['seed']}/latest.pt"
    before = file_hash(checkpoint)
    codec = create_legacy_codec(config).eval()
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    load_trainable(codec, artifact["model"])
    objective = ReconstructionLoss(config["stage1"]["loss"]).cuda()
    captured, rows, gradients = {}, [], []
    hooks = capture_encoders(codec, captured)
    available, _ = sources(config, "validation")
    try:
        for video in config["stage1"]["validation"]["videos"]:
            clip = read_window(source_by_id(available, video), 0, 2)
            state = ReferenceState()
            for frame_id, frame in enumerate(clip):
                rgb = frame[None].cuda().float() / 255
                with torch.no_grad():
                    encoded = codec.encode_frame(rgb, state, frame_id)
                measured, norms, state = diagnostic_frame(
                    codec, encoded, state, captured, rgb, objective, frame_id, video
                )
                rows.extend(measured)
                gradients.append({"video": video, "frame": frame_id, "budget": 4096, "terms": norms})
    finally:
        for hook in hooks:
            hook.remove()
    result = {
        "seed": config["seed"],
        "rows": rows,
        "summary": summarize(rows),
        "loss_gradients": gradients,
        "checkpoint_unchanged": before == file_hash(checkpoint),
        "source": "first two validation frames per video; same history for direct and token controls",
        "native_encoder_calls": codec.encode_calls,
        "note": "native targets captured by hooks from the one encoder call; diagnostic RGB controls never commit",
    }
    write_json(directory / "diagnostics.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/aip_vtok/gate1.yaml")
    parser.add_argument("--seed", type=int, choices=(1, 2), required=True)
    args = parser.parse_args()
    result = diagnose(settings(args.config, args.seed))
    print(json.dumps(result["summary"]), flush=True)


if __name__ == "__main__":
    main()
