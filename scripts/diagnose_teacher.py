"""Measure exact native targets under CP's different causal history, without approximate teachers."""

import torch
from torch.nn import functional as F

from cp_adaptok.backbones.intra_adapter import prepare_rgb
from cp_adaptok.backbones.native_baseline import NativeLatentTeacher
from cp_adaptok.data import read_video
from cp_adaptok.diagnostics import distribution_stats, gradient_norms
from cp_adaptok.runtime import build_model, metadata, read_config, resolve, seed_everything, write_json
from cp_adaptok.streaming.sender import ids_digest


def main():
    config = read_config("configs/pretrained_s1.yaml")
    seed_everything(config["seed"], config["device"], 0.65)
    model, _, _ = build_model(
        config, config["device"], checkpoint_override="checkpoints/pretrained_s1/step100.pt"
    )
    model.train()
    model.tokenizer.requires_grad_(True)
    model.vq.requires_grad_(True)
    model.decoder.requires_grad_(True)
    teacher = NativeLatentTeacher(
        resolve(config["paths"]["ld_checkpoint"]),
        resolve(config["paths"]["intra_checkpoint"]),
        model.device,
        model.config.qp,
    )
    clip = read_video(dict(config["data"]["train"][0], frames=4), 256, torch.Generator().manual_seed(1))
    state, records = model.empty_state(), []
    for t, frame in enumerate(clip):
        rgb = frame[None].to(model.device)
        target = teacher.target(rgb)
        encoded = model.encode_full(rgb, state)
        adapter = model.intra if t == 0 else model.ld
        cp_y = adapter.encode_latent(prepare_rgb(rgb), encoded.context)
        latent = model.decoder(
            encoded.vq.embeddings, encoded.context.latent_shape, encoded.context.frame_type
        )
        rec, _ = model.decode_latent(latent, encoded.context)
        image_loss, teacher_loss = F.mse_loss(rec, rgb), F.mse_loss(latent, target)
        row = {
            "frame": t,
            "frame_type": encoded.context.frame_type,
            "native_target": distribution_stats(target.flatten(2).transpose(1, 2)),
            "cp_encoder": distribution_stats(cp_y.flatten(2).transpose(1, 2)),
            "native_target_cp_encoder_mse": F.mse_loss(target, cp_y).item(),
            "teacher_mse": teacher_loss.item(),
            "reconstruction_mse": image_loss.item(),
            "reference_conditions_match": t == 0,
            "loss_gradients": [],
        }
        for weight in [0.0, 0.1, 1.0, 10.0]:
            model.zero_grad(set_to_none=True)
            (image_loss + weight * teacher_loss).backward(retain_graph=True)
            row["loss_gradients"].append({"teacher_weight": weight, **gradient_norms(model)})
        with torch.no_grad():
            _, feature = model.decode_ids(encoded.prefix(model.config.k_ref), encoded.context)
            state = encoded.context.commit(feature, ids_digest(encoded.prefix(model.config.k_ref))).detach()
        records.append(row)
    write_json(
        "results/optimization_20261115/teacher_conditions.json",
        dict(metadata(config, "diagnostic"), records=records, matched_p_teacher_verified=False),
    )


if __name__ == "__main__":
    main()
