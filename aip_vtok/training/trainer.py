"""Independent RNG ownership and recoverable short projector fitting."""

import time

import torch
from torch.nn import functional as F

from aip_vtok.backbones.latent_to_ref import LatentToRef
from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.runtime import resolve, save_checkpoint, write_json
from aip_vtok.training.checkpoint import require_identity, restore_training_state
from aip_vtok.training.data import prefetched_windows, sources, training_plans
from aip_vtok.training.determinism import configure_determinism


class RNGStreams:
    names = ("data_rng", "aug_rng", "budget_rng", "crop_rng")

    def __init__(self, seed):
        for index, name in enumerate(self.names):
            setattr(self, name, torch.Generator().manual_seed(seed + 1009 * index))

    def state_dict(self):
        return {name: getattr(self, name).get_state().clone() for name in self.names}

    def load_state_dict(self, states):
        for name in self.names:
            getattr(self, name).set_state(states[name].cpu())


def projector_sequence(backbone, projector, clip, flip=False):
    state, losses, diagnostics = ReferenceState(), [], []
    for frame_id, frame in enumerate(clip):
        rgb = frame[None].cuda().float() / 255
        if flip:
            rgb = rgb.flip(-1)
        with torch.no_grad():
            context = backbone.begin_frame(state, frame_id)
            latent = backbone.aligned_latent(rgb, context)
            _, target = backbone.decode_aligned(latent, context)
        predicted = projector(latent, context.intra)
        loss = F.mse_loss(predicted, target) / target.square().mean().clamp_min(1e-4)
        losses.append(loss)
        diagnostics.append(
            {
                "intra": context.intra,
                "relative_feature_mse": loss.detach().item(),
                "target_rms": target.square().mean().sqrt().item(),
                "predicted_rms": predicted.detach().square().mean().sqrt().item(),
            }
        )
        # Target and input share precisely this projector-owned history/context.
        state = context.commit_reference_once(predicted)
    return torch.stack(losses).mean(), diagnostics


def training_artifact(step, projector, optimizer, rng, plans, history, config, hashes):
    return {
        "step": step,
        "projector": projector.state_dict(),
        "width": projector.width,
        "optimizer": optimizer.state_dict(),
        "rng": rng.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "plans": plans,
        "history": history,
        "config": config,
        "manifest_hashes": hashes,
        "reference_source": "projector-only; target native Decoder has the identical frozen context",
    }


def restore_projector(checkpoint, projector, optimizer, rng, config, hashes):
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
    require_identity(artifact, config, hashes, hash_key="manifest_hashes")
    projector.load_state_dict(artifact["projector"])
    restore_training_state(artifact, optimizer, rng, stream_key="rng")
    return artifact["step"], artifact["plans"], artifact["history"]


def require_finite_projector_loss(loss, directory, step, diagnostics):
    if not torch.isfinite(loss):
        write_json(directory / "nonfinite_failure.json", {"step": step, "diagnostics": diagnostics})
        raise FloatingPointError("nonfinite projector loss")


def fit_projector(config, backbone, width, resume=False):
    configure_determinism()
    rows, hashes = sources(config, "train")
    cfg = config["projector_training"]
    directory = resolve(config["output"]) / f"projector_w{width}"
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve(config["checkpoints"]) / f"projector_w{width}/latest.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError(f"use --resume to continue {checkpoint}")
    projector = LatentToRef(width).cuda().to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(projector.parameters(), lr=cfg["lr"], weight_decay=0)
    rng = RNGStreams(config["seed"])
    plans = training_plans(rows, cfg["steps"], cfg["sequence_length"], rng)
    start, history = 0, []
    if resume:
        start, plans, history = restore_projector(checkpoint, projector, optimizer, rng, config, hashes)
    backbone.requires_grad_(False)
    started = time.perf_counter()
    for step, clip in enumerate(prefetched_windows(rows, plans[start:]), start + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, diagnostics = projector_sequence(backbone, projector, clip, plans[step - 1]["flip"])
        require_finite_projector_loss(loss, directory, step, diagnostics)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(projector.parameters(), 5.0, error_if_nonfinite=True)
        optimizer.step()
        row = {
            "step": step,
            "loss": loss.item(),
            "grad_norm": float(grad_norm),
            "video": rows[plans[step - 1]["source"]]["content_id"],
            "start": plans[step - 1]["start"],
            "diagnostics": diagnostics,
        }
        history.append(row)
        if step % 10 == 0 or step == 1:
            print(
                f"projector width={width} step={step}/{cfg['steps']} loss={loss.item():.6f} "
                f"elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )
        if step % cfg["save_every"] == 0 or step == cfg["steps"]:
            save_checkpoint(
                checkpoint, training_artifact(step, projector, optimizer, rng, plans, history, config, hashes)
            )
            write_json(
                directory / "training.json", {"history": history, "steps": step, "manifest_hashes": hashes}
            )
    return projector.eval()
