import json
import random
import time

import torch
from torch.nn import functional as F

from cp_adaptok.backbones.native_baseline import NativeLatentTeacher
from cp_adaptok.data import check_splits, read_video, sample_training_spec
from cp_adaptok.diagnostics import gradient_norms, vq_diagnostics
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
    monotonic = (
        torch.stack([F.relu(b - a) for a, b in zip(distortions, distortions[1:])]).mean()
        if len(budgets) > 1
        else rgb.new_zeros(())
    )
    estimated = (
        model.vq.expected_bits(encoded.features, model.entropy_model.logits, model.config.packet_tokens)
        if weights["estimated_rate"]
        else encoded.features.new_zeros(encoded.features.shape[:2])
    )
    pixels = rgb.shape[-1] * rgb.shape[-2]
    rate = torch.stack([estimated[:, :k].sum(-1).mean() / pixels for k in budgets]).mean()
    loss = loss + weights["monotonic"] * monotonic + weights["estimated_rate"] * rate
    records = [
        {
            "tokens": k,
            "mse": value.detach().item(),
            "estimated_bpp": estimated[:, :k].sum().detach().item() / pixels,
            "teacher_mse": F.mse_loss(latents[i].float(), teacher_target.float()).detach().item()
            if teacher_target is not None
            else None,
            "latent_delta_mse": F.mse_loss(latents[i].float(), latents[i - 1].float()).detach().item()
            if i
            else None,
            "total_loss": loss.detach().item(),
        }
        for i, (k, value) in enumerate(zip(budgets, distortions))
    ]
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


def train_clip(model, clip, budgets, config, perceptual, teacher, gradient_scale, diagnostic=False):
    state = model.empty_state()
    chunk_losses, records = [], []
    tbptt = config["training"]["tbptt"]
    if teacher:
        teacher.reset()
    for t, original in enumerate(clip):
        rgb = original[None].to(model.device)
        frame_type = "I" if t == 0 else "P"
        supervised = teacher is not None and teacher_applies(
            config.get("teacher", {"policy": "i_only"}), frame_type
        )
        target = teacher.target(rgb) if supervised else None
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
            digest = (
                ids_digest(encoded.prefix(model.config.k_ref))
                if encoded.vq.ids is not None
                else "continuous-diagnostic"
            )
            state = encoded.context.commit(feature, digest)
        chunk_losses.append(loss / len(clip) * gradient_scale)
        records.append(
            {
                "frame": t,
                "budgets": labels,
                "vq_usage": encoded.vq.usage.item(),
                "vq_perplexity": encoded.vq.perplexity.item(),
                "teacher": dict(
                    teacher.last_provenance,
                    reference_condition="independent_native_history"
                    if frame_type == "P"
                    else "intra_no_history",
                )
                if supervised
                else None,
                "frame_type": frame_type,
                "diagnostics": vq_diagnostics(model.vq, encoded.features, encoded.vq) if diagnostic else None,
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
    if "vq_training_state" in artifact:
        model.vq.restore_training_state(artifact["vq_training_state"])
    generator.set_state(artifact["sampling_rng"])
    torch.set_rng_state(artifact["torch_rng"])
    random.setstate(artifact["python_rng"])
    if artifact["cuda_rng"] and model.device.type == "cuda":
        torch.cuda.set_rng_state_all(artifact["cuda_rng"])
    return artifact["step"]


def teacher_applies(settings, frame_type):
    policy = settings.get("policy", "i_only")
    if policy == "legacy_ip":
        if not settings.get("allow_mismatched_p_diagnostic", False):
            raise ValueError("legacy I/P teacher is a mismatched-history diagnostic only")
        return True
    if policy not in ("none", "i_only"):
        raise ValueError("P-frame teacher needs a numerically verified same-context target")
    return policy == "i_only" and frame_type == "I"


def temperature_at(settings, step):
    initial = settings.get("temperature", 1.0)
    final = settings.get("final_temperature", initial)
    progress = min(1.0, step / max(1, settings.get("anneal_steps", 1)))
    return initial * (final / initial) ** progress


def set_quantization_step(model, config, step):
    settings = config["training"].get("vq", {})
    strategy = settings.get("strategy", "deterministic_nearest")
    if step < settings.get("warmup_steps", 0):
        strategy = "continuous"
    model.vq.set_training_policy(strategy, temperature_at(settings, step), settings.get("sampling_topk", 0))


def make_teacher(model, config):
    settings = config.get("teacher", {"policy": "i_only"})
    enabled = teacher_applies(settings, "I")
    if config["loss"].get("latent", 0) <= 0 or config.get("stage") == 3 or not enabled:
        return None
    paths = config["paths"]
    return NativeLatentTeacher(
        resolve(paths["ld_checkpoint"]), resolve(paths["intra_checkpoint"]), model.device, model.config.qp
    )


def optimization_step(model, optimizer, generator, step, config, perceptual, teacher, fixed_samples=None):
    train = config["training"]
    teacher_settings = config.get("teacher", {})
    if teacher_settings.get("decay_steps"):
        factor = max(teacher_settings.get("minimum_factor", 0.0), 1 - step / teacher_settings["decay_steps"])
        config = dict(config, loss=dict(config["loss"], latent=config["loss"]["latent"] * factor))
    optimizer.zero_grad(set_to_none=True)
    set_quantization_step(model, config, step)
    candidates = tuple(train["budgets"])
    budgets = (
        list(candidates)
        if train.get("all_budgets") or len(candidates) == 1
        else sample_budgets(candidates, generator)
    )
    diagnostic = (
        step in train.get("diagnostic_steps", [0, 10, 50, 100])
        or step % train.get("diagnostic_every", 100) == 0
    )
    records = []
    for _ in range(train["gradient_accumulation"]):
        index = int(torch.randint(len(config["data"]["train"]), (), generator=generator))
        spec, clip = training_sample(config, index, step, generator, fixed_samples)
        sampled = train_clip(
            model, clip, budgets, config, perceptual, teacher, 1 / train["gradient_accumulation"], diagnostic
        )
        for record in sampled:
            record.update(
                source=spec.path,
                source_frame=spec.start + record["frame"],
                input_shape=list(clip.shape[-2:]),
                quantization=model.vq.strategy,
                temperature=model.vq.temperature,
            )
        records += sampled
    gradients = gradient_norms(model) if diagnostic else None
    norm = torch.nn.utils.clip_grad_norm_(
        [p for p in model.parameters() if p.requires_grad], train["clip_grad_norm"]
    )
    if not torch.isfinite(norm):
        raise RuntimeError("nonfinite gradient norm; checkpoint remains at last completed update")
    optimizer.step()
    recovered = model.vq.finish_update(optimizer)
    records[-1].update(gradient_norms=gradients, recovered_codes=recovered)
    return budgets, records, norm.item()


def training_sample(config, index, step, generator, fixed_samples=None):
    train = config["training"]
    length, crop = sequence_length(train, step), resolution_at(train, step)
    key = index, length, str(crop)
    if fixed_samples is not None and key in fixed_samples:
        return fixed_samples[key]
    rng = (
        torch.Generator().manual_seed(train["fixed_sample_seed"])
        if "fixed_sample_seed" in train
        else generator
    )
    spec = sample_training_spec(config["data"]["train"][index], length, rng)
    sample = spec, read_video(spec, crop, rng)
    if fixed_samples is not None:
        fixed_samples[key] = sample
    return sample


def resolution_at(training, step):
    crop = training["crop"]
    for point in training.get("resolution_schedule", []):
        if step >= point["step"]:
            crop = point["crop"]
    return crop


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
    model.vq.configure_dead_codes(**train.get("vq", {}).get("dead_codes", {}))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=train["lr"])
    generator = torch.Generator().manual_seed(config["seed"])
    output = resolve(config["paths"]["output"])
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = resolve(config["paths"]["save_checkpoint"])
    perceptual, teacher = perceptual_networks(config, model.device), make_teacher(model, config)
    model.train()
    _ = model.model_id
    # Auxiliary constructors consume RNG while initializing their pretrained weights.
    step = restore_training(model, optimizer, generator, artifact, config["stage"]) if resume else 0
    start = time.perf_counter()
    fixed_samples = {} if "fixed_sample_seed" in train else None
    try:
        while step < max_steps:
            vq_settings = train.get("vq", {})
            if vq_settings.get("initialization") and step == vq_settings.get("warmup_steps", 0):
                from cp_adaptok.training_initialization import initialize_from_training_data

                initialize_from_training_data(model, config, output)
            budgets, records, norm = optimization_step(
                model, optimizer, generator, step, config, perceptual, teacher, fixed_samples
            )
            step += 1
            record_update(output, step, start, norm, records, budgets, status)
            if step % train["save_every"] == 0 or step == max_steps:
                save_checkpoint(checkpoint, model, optimizer, generator, step, config, status)
            if step % train["validate_every"] == 0 and model.vq.strategy != "continuous":
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
