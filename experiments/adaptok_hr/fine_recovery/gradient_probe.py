"""Supplement E0 with gradients through the actual discrete-forward STE and restore maps."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from adaptok_hr.codec.codec import Codec
from adaptok_hr.data.dataset import read_clip
from adaptok_hr.data.manifest import load_index
from adaptok_hr.models.temporal_state import frame_context
from adaptok_hr.runtime import environment, file_hash, seed_everything, write_json
from adaptok_hr.training.checkpoint import load_model
from cp_adaptok.metrics import MetricSuite

from .diagnose import LEVELS, layer_slices, projection_tokens
from .evaluate import VALIDATION_IDS


def objectives(rgb, target, perceptual):
    mse = F.mse_loss(rgb.float(), target)
    trained = mse + F.l1_loss(rgb.float(), target) + .1 * perceptual(rgb.float() * 2 - 1, target * 2 - 1).mean()
    return {"rgb_mse": mse, "training_reconstruction": trained}


def token_gradients(model, tokens, discrete, reference, target, perceptual):
    leaf = tokens.detach().requires_grad_()
    z = F.normalize(leaf, dim=-1)
    ste = discrete.detach() + (z - z.detach())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rgb, _ = model.decoder(ste[:, :2048], reference)
    losses = objectives(rgb, target, perceptual)
    result = {}
    for name, loss in losses.items():
        grad, = torch.autograd.grad(loss, leaf, retain_graph=True)
        result[name] = {level: float(grad[:, span].norm()) for level, span in layer_slices(model, 2048)}
        result[name]["untransmitted_tail_norm"] = float(grad[:, 2048:].norm())
    return result


def map_gradients(model, discrete, reference, target, perceptual):
    maps = tuple(x.detach().requires_grad_() for x in model.positions.scatter(discrete[:, :2048]))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        feature = model.decoder.fuse_maps(maps, reference)
        rgb = model.decoder.upsample_rgb(feature, reference)
    losses = objectives(rgb, target, perceptual)
    result = {}
    for name, loss in losses.items():
        grads = torch.autograd.grad(loss, maps, retain_graph=True)
        result[name] = {level: {"embedding_norm": float(grad[:, :-1].norm()),
                                "mask_norm": float(grad[:, -1:].norm())}
                        for level, grad in zip(LEVELS, grads)}
    return result


def main():
    checkpoint = "checkpoints/adaptok_hr/joint/step_001000.pt"
    output = Path("results/adaptok_hr_fine_recovery/E0/actual_ste_gradients.json")
    if output.exists():
        raise ValueError("gradient probe already exists")
    seed_everything(20261118)
    torch.set_num_threads(4)
    model, _ = load_model(checkpoint)
    model.requires_grad_(False)
    index = load_index("data/adaptok_hr/index.json")
    metric = MetricSuite("cuda", ("lpips",))
    if metric.errors:
        raise RuntimeError(metric.errors)
    perceptual = metric.networks["lpips"]
    codec = Codec(model, amp_bf16=True)
    rows = []
    for source in (r for r in index["splits"]["validation"] if r["content_id"] in VALIDATION_IDS):
        clip, _ = read_clip(source, 0, 8)
        state = None
        for number, cpu in enumerate(clip):
            target = cpu[None].cuda()
            if number in (0, 1, 7):
                _, _, reference = frame_context(state, model.config.gop)
                with torch.no_grad(), codec.autocast():
                    encoded, raw = projection_tokens(model, target, state)
                discrete = model.vq.lookup(encoded.ids).detach()
                rows.append({"content_id": source["content_id"], "frame": number,
                             "actual_vq_ste": token_gradients(model, F.normalize(raw, dim=-1), discrete, reference, target, perceptual),
                             "decoder_restore_maps": map_gradients(model, discrete, reference, target, perceptual)})
            _, state = codec.encode_frame(target, state, 2048, source["content_id"])
    write_json(output, {"environment": environment(), "checkpoint_sha256": file_hash(checkpoint),
                        "manifest_identity": index["identity"], "diagnostic_only": True, "rows": rows,
                        "training_reconstruction": "MSE + L1 + 0.1 LPIPS; excludes VQ, temporal and monotonic objectives"})
    print(json.dumps({"frames": len(rows)}))


if __name__ == "__main__":
    main()
