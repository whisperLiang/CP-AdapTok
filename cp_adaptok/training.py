import json
import random
import time

import torch
from torch.nn import functional as F

from cp_adaptok.backbones.native_baseline import NativeLatentTeacher
from cp_adaptok.data import check_splits, read_video
from cp_adaptok.runtime import metadata, resolve, save_checkpoint, write_json
from cp_adaptok.streaming.sender import ids_digest
from cp_adaptok.tokenizer.prefix_masking import sample_budgets


def configure_stage(model, stage, config):
    model.requires_grad_(False)
    if stage in (1, 2, 4):
        for part in (model.tokenizer, model.vq, model.decoder):
            part.requires_grad_(True)
    elif stage == 3:
        model.entropy_model.requires_grad_(True)
        model.scorer.requires_grad_(True)
    else:
        raise ValueError("train supports stages 1-4; use eval --baseline native for stage 0")
    if stage == 4:
        for name in config["training"]["fine_tune"]:
            model.get_submodule(name).requires_grad_(True)


def perceptual_networks(config, device):
    from cp_adaptok.metrics import MetricSuite

    names = [name for name in ("lpips", "dists") if config["loss"].get(name, 0) > 0]
    suite = MetricSuite(device, names)
    if suite.errors:
        raise RuntimeError(f"requested training loss unavailable: {suite.errors}")
    return suite.networks


def frame_objective(model, encoded, rgb, budgets, config, perceptual, teacher_target=None):
    weights = config["loss"]
    reconstructions, latents, distortions = [], [], []
    for k in budgets:
        latent = model.decoder(
            encoded.vq.embeddings[:, :k], encoded.context.latent_shape, encoded.context.frame_type
        )
        reconstruction, _ = model.decode_latent(latent, encoded.context)
        latents.append(latent)
        reconstructions.append(reconstruction)
        distortions.append(F.mse_loss(reconstruction, rgb))
    loss = weights["mse"] * torch.stack(distortions).mean() + weights["vq"] * encoded.vq.loss
    if teacher_target is not None:
        loss = (
            loss
            + weights["latent"]
            * torch.stack([F.mse_loss(y.float(), teacher_target.float()) for y in latents]).mean()
        )
    loss = loss + perceptual_objective(rgb, reconstructions, perceptual, weights)
    monotonic = torch.stack([F.relu(b - a) for a, b in zip(distortions, distortions[1:])]).mean()
    rate = model.entropy_model.loss(encoded.vq.ids.detach(), model.config.packet_tokens)
    loss = loss + weights["monotonic"] * monotonic + weights["estimated_rate"] * rate
    records = [{"tokens": k, "mse": value.detach().item()} for k, value in zip(budgets, distortions)]
    return loss, records


def perceptual_objective(rgb, reconstructions, perceptual, weights):
    loss = rgb.new_zeros(())
    for name, network in perceptual.items():
        values = [
            network(rgb * 2 - 1, rec * 2 - 1) if name == "lpips" else network(rgb, rec)
            for rec in reconstructions
        ]
        loss = loss + weights[name] * torch.stack([v.mean() for v in values]).mean()
    return loss


def auxiliary_objective(model, encoded, rgb, budgets, config):
    labels = []
    with torch.no_grad():
        for k in budgets:
            rec, _ = model.decode_ids(encoded.prefix(k), encoded.context)
            labels.append(F.mse_loss(rec, rgb))
    target = torch.stack(labels)[None]
    predicted = model.scorer(encoded.features.detach(), budgets)
    entropy = model.entropy_model.loss(encoded.vq.ids.detach(), model.config.packet_tokens)
    loss = config["loss"]["entropy"] * entropy + config["loss"]["scorer"] * F.mse_loss(predicted, target)
    records = [
        {
            "tokens": k,
            "mse": actual.item(),
            "predicted_mse": prediction.item(),
            "prediction_error": (prediction - actual).item(),
            "entropy_bits_per_token": entropy.item(),
        }
        for k, actual, prediction in zip(budgets, labels, predicted[0])
    ]
    return loss, records


def train_clip(model, clip, budgets, config, perceptual, teacher, gradient_scale):
    state = model.empty_state()
    chunk_losses, records = [], []
    tbptt = config["training"]["tbptt"]
    if teacher:
        teacher.reset()
    for t, original in enumerate(clip):
        rgb = original[None].to(model.device)
        target = teacher.target(rgb) if teacher else None
        with torch.autocast(
            device_type=model.device.type,
            dtype=torch.bfloat16,
            enabled=config["training"]["amp_bf16"] and model.device.type == "cuda",
        ):
            encoded = model.encode_full(rgb, state)
            if config["stage"] == 3:
                loss, labels = auxiliary_objective(model, encoded, rgb, budgets, config)
            else:
                loss, labels = frame_objective(model, encoded, rgb, budgets, config, perceptual, target)
            # Differentiable base-only loop, independent of sampled enhancement budgets.
            _, feature = model.decode_embeddings(
                encoded.vq.embeddings[:, : model.config.k_ref], encoded.context
            )
            state = encoded.context.commit(feature, ids_digest(encoded.prefix(model.config.k_ref)))
        chunk_losses.append(loss / len(clip) * gradient_scale)
        records.append(
            {
                "frame": t,
                "budgets": labels,
                "vq_usage": encoded.vq.usage.item(),
                "vq_perplexity": encoded.vq.perplexity.item(),
                "teacher": teacher.last_provenance if teacher else None,
            }
        )
        if (t + 1) % tbptt == 0 or t == len(clip) - 1:
            torch.stack(chunk_losses).sum().backward()
            chunk_losses = []
            state = state.detach()
    return records


def sequence_length(training, step):
    length = training["sequence_length"]
    for point in training.get("sequence_schedule", []):
        if step >= point["step"]:
            length = point["frames"]
    return length


def restore_training(model, optimizer, generator, artifact, stage):
    if artifact is None or artifact["config"].get("stage") != stage:
        raise ValueError("resume requires a checkpoint from the same training stage")
    optimizer.load_state_dict(artifact["optimizer"])
    generator.set_state(artifact["sampling_rng"])
    torch.set_rng_state(artifact["torch_rng"])
    random.setstate(artifact["python_rng"])
    if artifact["cuda_rng"] and model.device.type == "cuda":
        torch.cuda.set_rng_state_all(artifact["cuda_rng"])
    return artifact["step"]


def make_teacher(model, config):
    if config["loss"].get("latent", 0) <= 0:
        return None
    paths = config["paths"]
    return NativeLatentTeacher(
        resolve(paths["ld_checkpoint"]), resolve(paths["intra_checkpoint"]), model.device, model.config.qp
    )


def optimization_step(model, optimizer, generator, step, config, perceptual, teacher):
    train = config["training"]
    optimizer.zero_grad(set_to_none=True)
    budgets = sample_budgets(tuple(train["budgets"]), generator)
    records = []
    for _ in range(train["gradient_accumulation"]):
        index = int(torch.randint(len(config["data"]["train"]), (), generator=generator))
        spec = dict(config["data"]["train"][index], frames=sequence_length(train, step))
        clip = read_video(spec, train["crop"], generator)
        records += train_clip(
            model, clip, budgets, config, perceptual, teacher, 1 / train["gradient_accumulation"]
        )
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], train["clip_grad_norm"]
    )
    if not torch.isfinite(norm):
        raise RuntimeError("nonfinite gradient norm; checkpoint remains at last completed update")
    optimizer.step()
    return budgets, records, norm.item()


def record_update(output, step, start, norm, records, budgets, status):
    record = {
        "step": step,
        "elapsed_s": time.perf_counter() - start,
        "grad_norm": norm,
        "records": records,
        "status": status,
        "real_bitstream_rd": False,
    }
    with (output / "train.jsonl").open("a") as handle:
        handle.write(json.dumps(record) + "\n")
    if step == 1 or step % 10 == 0:
        print(json.dumps({"step": step, "grad_norm": norm, "budgets": budgets, "status": status}), flush=True)


def validate_training(model, config, step, output, status):
    from cp_adaptok.evaluation import evaluate_sequences

    model.eval()
    model.refresh_identity()
    validation = evaluate_sequences(
        model,
        config,
        split="validation",
        budgets=config["training"]["budgets"],
        metric_names=("psnr",),
        status=status,
        output=output / f"validation_{step}",
    )
    write_json(output / f"validation_{step}.json", validation)
    model.train()


def run_training(model, config, max_steps, artifact=None, resume=False, status="trained"):
    check_splits(config["data"])
    train = config["training"]
    configure_stage(model, config["stage"], config)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=train["lr"])
    generator = torch.Generator().manual_seed(config["seed"])
    step = restore_training(model, optimizer, generator, artifact, config["stage"]) if resume else 0
    output = resolve(config["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve(config["paths"]["save_checkpoint"])
    perceptual, teacher = perceptual_networks(config, model.device), make_teacher(model, config)
    model.train()
    _ = model.model_id
    start = time.perf_counter()
    try:
        while step < max_steps:
            budgets, records, norm = optimization_step(
                model, optimizer, generator, step, config, perceptual, teacher
            )
            step += 1
            record_update(output, step, start, norm, records, budgets, status)
            if step % train["save_every"] == 0 or step == max_steps:
                save_checkpoint(checkpoint, model, optimizer, generator, step, config, status)
            if step % train["validate_every"] == 0:
                validate_training(model, config, step, output, status)
    except torch.cuda.OutOfMemoryError:
        write_json(
            output / "oom.json",
            {
                "step": step,
                "memory": torch.cuda.memory_summary(),
                "resume_checkpoint": str(checkpoint),
                "suggestions": ["reduce crop", "reduce tbptt", "increase accumulation"],
            },
        )
        raise
    report = metadata(config, status)
    report.update(
        completed_steps=step,
        checkpoint=str(checkpoint),
        trained_rd_validated=False,
        elapsed_s=time.perf_counter() - start,
    )
    write_json(output / "training_report.json", report)
    return report
