"""Measure frozen Fine directions and separate gradients on the same native Train I frame."""

import argparse
import json

import torch

from adaptok_hr.data.dataset import read_clip
from adaptok_hr.data.manifest import load_index
from adaptok_hr.models.vq import code_statistics
from adaptok_hr.runtime import environment, file_hash, seed_everything, write_json
from adaptok_hr.training.checkpoint import load_model
from adaptok_hr.training.fine_detail import suffix_supervision
from adaptok_hr.training.losses import ReconstructionLoss
from experiments.adaptok_hr.fine_recovery.diagnose import feature_statistics, projection_tokens

from .run import PROTOCOL, ROOT, checkpoint


def measure(path, frame, losses):
    model, saved = load_model(path)
    model.requires_grad_(False)
    inputs = []
    handle = model.vq.register_forward_pre_hook(lambda module, args: inputs.append(args[0].detach()))
    try:
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            actual, raw = projection_tokens(model, frame, None)
    finally:
        handle.remove()
    leaf = inputs[0].requires_grad_()
    encoded = model.vq(leaf)
    if not torch.equal(encoded.ids, actual.ids):
        raise RuntimeError("gradient probe does not reproduce the actual transmitted IDs")
    low, high = PROTOCOL["information"]["suffix"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = model.forward_frame(frame, None, high, encoded)
        distortion, _ = losses.distortion(output.rgb, frame)
        weight = saved["config"]["training"]["losses"].get("suffix_detail", 0.)
        supervised = suffix_supervision(model, frame, None, encoded, {high: output}, weight)
    rec, = torch.autograd.grad(distortion, leaf, retain_graph=True)
    vq, = torch.autograd.grad(encoded.loss_terms(high)["vq_loss"], leaf, retain_graph=True)
    aux = torch.autograd.grad(supervised, leaf)[0] if weight else torch.zeros_like(leaf)
    row = {"step": saved["completed_steps"], "checkpoint_sha256": file_hash(path), "actual_id_match": True,
           "reconstruction_loss": float(distortion.detach()), "weighted_suffix_loss": float(supervised.detach()),
           "unsent_reconstruction_gradient": float(rec[:, high:].norm()), "prefix_suffix_gradient": float(aux[:, :low].norm()),
           "unsent_suffix_gradient": float(aux[:, high:].norm()), "levels": {}}
    offset = sum(model.positions.counts[:2])
    for name, span in (("transmitted_fine", slice(offset, high)), ("incremental_suffix", slice(low, high))):
        row["levels"][name] = {**code_statistics(encoded.ids[:, span]),
                               "projection": feature_statistics(raw[:, span]), "vq_input": feature_statistics(leaf[:, span]),
                               "reconstruction_gradient_norm": float(rec[:, span].norm()),
                               "weighted_suffix_gradient_norm": float(aux[:, span].norm()), "vq_gradient_norm": float(vq[:, span].norm())}
    return row


def run(names, steps):
    seed_everything(20261118)
    torch.set_num_threads(4)
    index = load_index("data/adaptok_hr/index.json")
    source = next(r for r in index["splits"]["train"] if r["content_id"] == "DAVIS2017:bear")
    clip, input_meta = read_clip(source, 0, 1)
    losses = ReconstructionLoss({"l1": 1., "mse": 1., "perceptual": .1}, "cuda")
    for name in names:
        for step in steps:
            dst = ROOT / name / "probes" / f"{step:06d}.json"
            if dst.exists():
                continue
            result = measure(checkpoint(name, step), clip.cuda(), losses)
            write_json(dst, {"diagnostic_only": True, "optimizer_updates": 0, "input": input_meta,
                             "manifest_identity": index["identity"], "environment": environment(), **result})
            print(json.dumps({"name": name, "step": step, "codes": result["levels"]["incremental_suffix"]["used_codes"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[0, 10, 25, 100, 300])
    args = parser.parse_args()
    run(args.names, args.steps)
