import json
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from cp_adaptok.backbones.native_baseline import NativeLatentTeacher
from cp_adaptok.data import check_splits, read_video, sample_training_spec
from cp_adaptok.diagnostics import gradient_norms, vq_diagnostics
from cp_adaptok.runtime import metadata, resolve, save_checkpoint, write_json
from cp_adaptok.streaming.sender import ids_digest
from cp_adaptok.tokenizer.prefix_masking import sample_budgets, sample_nested_budgets


def active_budget_weights(budgets, settings, step=0):
    weights = settings.get("weights")
    for point in settings.get("weight_schedule", []):
        if step >= point["step"]:
            weights = point["weights"]
    if weights is None:
        return [1.0] * len(budgets)
    values = [float(weights.get(k, weights.get(str(k), float("nan")))) for k in budgets]
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise ValueError("every budget needs an explicit finite positive weight")
    return values


def prefix_distortion(distortions, budgets, settings, step=0):
    weights = active_budget_weights(budgets, settings, step)
    return torch.stack([weight * value for weight, value in zip(weights, distortions)]).mean()


def progressive_penalty(distortions, settings):
    pairs = zip(distortions, distortions[1:])
    if settings.get("mode") == "progressive_nested" and settings.get("one_sided", True):
        pairs = ((low.detach(), high) for low, high in pairs)
    values = [F.relu(high - low + settings.get("margin", 0.0)) for low, high in pairs]
    return torch.stack(values).mean() if values else distortions[0].new_zeros(())


def training_budgets(model, config, generator):
    train, settings = config["training"], config.get("prefix_training", {})
    candidates = list(settings.get("budgets", train["budgets"]))
    if settings.get("mode") == "progressive_nested":
        return sample_nested_budgets(candidates, generator, settings.get("sample_intermediate", 1))
    return candidates if train.get("all_budgets") or len(candidates) == 1 else sample_budgets(candidates, generator)


def validate_prefix_training(model, config):
    from cp_adaptok.streaming.packetizer import validate_budgets

    settings = config.get("prefix_training", {})
    if not settings:
        return
    if settings.get("mode") not in ("legacy", "progressive_nested"):
        raise ValueError("unknown prefix training mode")
    budgets = settings.get("budgets", config["training"]["budgets"])
    validate_budgets(budgets, model.config)
    if budgets != config["training"]["budgets"] or budgets[0] != model.config.k_ref or budgets[-1] != model.config.n_max:
        raise ValueError("training budgets must span K_ref to N_max consistently")
    if not settings.get("always_include_base", True) or not settings.get("always_include_max", True):
        raise ValueError("progressive training requires both endpoints")
    if settings.get("margin", 0) < 0:
        raise ValueError("progress margin must be nonnegative")
    active_budget_weights(budgets, settings)


def prefix_gradient_records(distortions, embeddings, packet_tokens):
    from cp_adaptok.diagnostics_prefix import gradient_groups

    return [gradient_groups(torch.autograd.grad(value, embeddings, retain_graph=True)[0], packet_tokens)
            for value in distortions]


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


def frame_objective(model, encoded, rgb, budgets, config, perceptual, teacher_target=None, diagnostic=False):
    weights = config["loss"]
    settings = config.get("prefix_training", {})
    step = config.get("prefix_step", 0)
    reconstructions, latents, distortions = [], [], []
    for k in budgets:
        latent = model.decoder(
            encoded.vq.embeddings[:, :k], encoded.context.latent_shape, encoded.context.frame_type
        )
        reconstruction, _ = model.decode_latent(latent, encoded.context)
        latents.append(latent)
        reconstructions.append(reconstruction)
        distortions.append(F.mse_loss(reconstruction, rgb))
    loss = weights["mse"] * prefix_distortion(distortions, budgets, settings, step) + weights["vq"] * encoded.vq.loss
    if teacher_target is not None:
        loss = (
            loss
            + weights["latent"]
            * prefix_distortion([F.mse_loss(y.float(), teacher_target.float()) for y in latents],
                                budgets, settings, step)
        )
    loss = loss + perceptual_objective(rgb, reconstructions, perceptual, weights)
    monotonic = progressive_penalty(distortions, settings)
    estimated = (
        model.vq.expected_bits(encoded.features, model.entropy_model.logits, model.config.packet_tokens)
        if weights["estimated_rate"]
        else encoded.features.new_zeros(encoded.features.shape[:2])
    )
    pixels = rgb.shape[-1] * rgb.shape[-2]
    rate = torch.stack([estimated[:, :k].sum(-1).mean() / pixels for k in budgets]).mean()
    loss = loss + weights["monotonic"] * monotonic + weights["estimated_rate"] * rate
    records = objective_records(budgets, distortions, estimated, pixels, latents, teacher_target, loss)
    budget_weights = active_budget_weights(budgets, settings, step)
    for row, weight in zip(records, budget_weights):
        row.update(distortion_weight=weight, progressive_penalty=monotonic.detach().item())
    if diagnostic and settings.get("monitor_gradients", False):
        gradients = prefix_gradient_records(distortions, encoded.vq.embeddings, model.config.packet_tokens)
        for row, groups in zip(records, gradients):
            row["embedding_gradient_by_group"] = groups
    return loss, records


def objective_records(budgets, distortions, estimated, pixels, latents, teacher_target, loss):
    return [
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
    model.tokenizer.diagnostics_enabled = diagnostic
    if teacher:
        teacher.reset()
    for t, original in enumerate(clip):
        rgb = original[None].to(model.device)
        frame_type = "I" if t == 0 else "P"
        target = teacher_target(teacher, config, frame_type, rgb)
        with torch.autocast(
            device_type=model.device.type,
            dtype=torch.bfloat16,
            enabled=config["training"]["amp_bf16"] and model.device.type == "cuda",
        ):
            encoded = model.encode_full(rgb, state)
            if config["stage"] == 3:
                loss, labels = auxiliary_objective(model, encoded, rgb, budgets, config)
            else:
                loss, labels = frame_objective(model, encoded, rgb, budgets, config, perceptual, target, diagnostic)
            # Differentiable base-only loop, independent of sampled enhancement budgets.
            _, feature = model.decode_embeddings(
                encoded.vq.embeddings[:, : model.config.k_ref], encoded.context
            )
            digest = training_digest(encoded, model.config.k_ref)
            state = encoded.context.commit(feature, digest)
        chunk_losses.append(loss / len(clip) * gradient_scale)
        records.append(
            frame_record(model, encoded, labels, teacher if target is not None else None, t, diagnostic)
        )
        if (t + 1) % tbptt == 0 or t == len(clip) - 1:
            torch.stack(chunk_losses).sum().backward()
            chunk_losses = []
            state = state.detach()
    return records


def teacher_target(teacher, config, frame_type, rgb):
    supervised = teacher is not None and config["loss"].get("latent", 0) > 0
    if supervised and teacher_applies(config.get("teacher", {"policy": "i_only"}), frame_type):
        return teacher.target(rgb)
    return None


def training_digest(encoded, k_ref):
    return ids_digest(encoded.prefix(k_ref)) if encoded.vq.ids is not None else "continuous-diagnostic"


def frame_record(model, encoded, labels, teacher, t, diagnostic):
    frame_type = encoded.context.frame_type
    provenance = dict(teacher.last_provenance) if teacher else None
    if provenance is not None:
        provenance["reference_condition"] = (
            "independent_native_history" if frame_type == "P" else "intra_no_history"
        )
    layers = getattr(model.tokenizer, "last_diagnostics", {})
    return {
        "frame": t,
        "budgets": labels,
        "vq_usage": encoded.vq.usage.item(),
        "vq_perplexity": encoded.vq.perplexity.item(),
        "teacher": provenance,
        "frame_type": frame_type,
        "diagnostics": vq_diagnostics(model.vq, encoded.features, encoded.vq) if diagnostic else None,
        "query_layers": dict(layers) if diagnostic else None,
    }


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
    if "numpy_rng" in artifact:
        rng = artifact["numpy_rng"]
        np.random.set_state((rng[0], np.array(rng[1], dtype=np.uint32), *rng[2:]))
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
    if config.get("stage") == 3:
        strategy = "deterministic_nearest"
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
    config = dict(decayed_teacher_config(config, step), prefix_step=step)
    optimizer.zero_grad(set_to_none=True)
    set_quantization_step(model, config, step)
    budgets = training_budgets(model, config, generator)
    diagnostic = diagnostic_step(train, step)
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


def decayed_teacher_config(config, step):
    settings = config.get("teacher", {})
    if not settings.get("decay_steps"):
        return config
    factor = max(settings.get("minimum_factor", 0.0), 1 - step / settings["decay_steps"])
    return dict(config, loss=dict(config["loss"], latent=config["loss"]["latent"] * factor))


def diagnostic_step(train, step):
    return (
        step in train.get("diagnostic_steps", [0, 10, 50, 100])
        or step % train.get("diagnostic_every", 100) == 0
    )


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
    validate_prefix_training(model, config)
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
    initialize_queries(model, train, resume)
    _ = model.model_id
    # Auxiliary constructors consume RNG while initializing their pretrained weights.
    step = restore_training(model, optimizer, generator, artifact, config["stage"]) if resume else 0
    start = time.perf_counter()
    fixed_samples = {} if "fixed_sample_seed" in train else None
    try:
        step = training_loop(
            model,
            optimizer,
            generator,
            config,
            step,
            max_steps,
            perceptual,
            teacher,
            fixed_samples,
            output,
            checkpoint,
            start,
            status,
        )
    except torch.cuda.OutOfMemoryError:
        write_json(
            output / "oom.json",
            {
                "started_step": step,
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


def initialize_queries(model, train, resume):
    if resume:
        return
    if train.get("query_initialization_scale"):
        model.tokenizer.rescale_initial_queries(train["query_initialization_scale"])
    if train.get("grid_query_initialization_scale"):
        with torch.no_grad():
            model.decoder.projector.query.mul_(train["grid_query_initialization_scale"])


def training_loop(
    model,
    optimizer,
    generator,
    config,
    step,
    max_steps,
    perceptual,
    teacher,
    fixed_samples,
    output,
    checkpoint,
    start,
    status,
):
    train = config["training"]
    while step < max_steps:
        settings = train.get("vq", {})
        if settings.get("initialization") and step == settings.get("warmup_steps", 0):
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
    return step
