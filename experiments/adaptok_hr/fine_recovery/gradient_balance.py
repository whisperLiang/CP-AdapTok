"""Post-pilot mechanism probe: measure gradient balance, without a training update."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from adaptok_hr.data.dataset import read_clip
from adaptok_hr.data.manifest import load_index
from adaptok_hr.models.vq import code_statistics
from adaptok_hr.runtime import environment, file_hash, seed_everything, write_json
from adaptok_hr.training.checkpoint import load_model
from cp_adaptok.metrics import MetricSuite

from .diagnose import feature_statistics, layer_slices, projection_tokens


def measure(model, target, perceptual):
    model.requires_grad_(False)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        _, raw = projection_tokens(model, target, None)
    leaf = F.normalize(raw, dim=-1).detach().requires_grad_()
    encoded = model.vq(leaf)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        rgb, _ = model.decoder(encoded.quantized[:, :2048], None)
    reconstruction = F.mse_loss(rgb.float(), target) + F.l1_loss(rgb.float(), target)
    reconstruction += .1 * perceptual(rgb.float() * 2 - 1, target * 2 - 1).mean()
    vq = encoded.loss_terms(2048)["vq_loss"]
    reconstruction_grad, = torch.autograd.grad(reconstruction, leaf, retain_graph=True)
    vq_grad, = torch.autograd.grad(vq, leaf)
    levels = {}
    for name, span in layer_slices(model, 2048):
        rec, commit = float(reconstruction_grad[:, span].norm()), float(vq_grad[:, span].norm())
        levels[name] = {**code_statistics(encoded.ids[:, span]), "raw": feature_statistics(raw[:, span]),
                        "normalized": feature_statistics(leaf[:, span]),
                        "reconstruction_gradient_norm": rec, "vq_encoder_gradient_norm": commit,
                        "vq_to_reconstruction_gradient_ratio": commit / max(rec, 1e-20)}
    return {"levels": levels, "reconstruction_loss": float(reconstruction.detach()), "vq_loss": float(vq.detach())}


def main():
    seed_everything(20261118)
    torch.set_num_threads(4)
    output = Path("results/adaptok_hr_fine_recovery/gradient_balance_300.json")
    if output.exists():
        raise ValueError("probe already measured")
    index = load_index("data/adaptok_hr/index.json")
    source = next(r for r in index["splits"]["train"] if r["content_id"] == "DAVIS2017:bear")
    clip, metadata = read_clip(source, 0, 1)
    target = clip.cuda()
    metrics = MetricSuite("cuda", ("lpips",))
    if metrics.errors:
        raise RuntimeError(metrics.errors)
    rows = []
    for name in ("A0", "A1", "A2", "A3"):
        for step in (0, 100, 300):
            checkpoint = f"checkpoints/adaptok_hr_fine_recovery/{name}/step_{step:06d}.pt"
            model, _ = load_model(checkpoint)
            row = {"name": name, "step": step, "checkpoint_sha256": file_hash(checkpoint),
                   **measure(model, target, metrics.networks["lpips"])}
            rows.append(row)
            print(json.dumps({"name": name, "step": step, "fine": row["levels"]["Fine"]}), flush=True)
            del model
    write_json(output, {"environment": environment(), "diagnostic_only": True, "optimizer_updates": 0,
                        "manifest_identity": index["identity"], "input": metadata, "rows": rows,
                        "scope": "single actual Train bear I-frame at K2048; frozen snapshots; loss uses recorded .25 commitment and 1.0 VQ weight, reconstruction MSE+L1+0.1LPIPS; no auxiliary, temporal or monotonic term; ratios are local, not training causal proof"})


if __name__ == "__main__":
    main()
