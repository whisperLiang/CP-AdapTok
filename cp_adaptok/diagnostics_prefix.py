"""Decoder-only counterfactuals and measured autograd signals, never network packets."""

import math

import torch
from torch.nn import functional as F

from cp_adaptok.diagnostics import distribution_stats
from cp_adaptok.metrics import MetricSuite
from cp_adaptok.streaming.sender import ids_digest


def group_slices(length, packet_tokens):
    if packet_tokens < 1 or length < 1 or length % packet_tokens:
        raise ValueError("diagnostics require nonempty packet-aligned prefixes")
    return [slice(first, first + packet_tokens) for first in range(0, length, packet_tokens)]


def mean_cosine(a, b=None):
    a = F.normalize(a.float().flatten(0, -2), dim=-1)
    if b is not None:
        b = F.normalize(b.float().flatten(0, -2), dim=-1)
        return (a.mean(0) @ b.mean(0)).item()
    count = len(a)
    return ((a.sum(0).square().sum() - a.square().sum()) / max(1, count * (count - 1))).item()


def js_divergence(a, b):
    middle = (a + b) / 2
    log_middle = middle.clamp_min(1e-30).log()
    return ((a * (a.clamp_min(1e-30).log() - log_middle)).sum()
            + (b * (b.clamp_min(1e-30).log() - log_middle)).sum()).item() / 2


@torch.no_grad()
def token_group_statistics(ids, content, positions, packet_tokens, codebook_size, previous=None):
    slices = group_slices(ids.shape[1], packet_tokens)
    probabilities, groups = [], []
    for index, part in enumerate(slices):
        values = content[:, part].float()
        counts = torch.bincount(ids[:, part].flatten(), minlength=codebook_size).float()
        probability = counts / counts.sum()
        probabilities.append(probability)
        row = {
            "group": index, "first": part.start, "end": part.stop,
            "unique_ids": int((counts > 0).sum()),
            "perplexity": (-(probability * probability.clamp_min(1e-30).log()).sum()).exp().item(),
            "content_variance": values.var(1, unbiased=False).mean().item(),
            "positioned_variance": (values + positions[:, part]).var(1, unbiased=False).mean().item(),
            "content_cosine": mean_cosine(values),
            "position_cosine": mean_cosine(positions[:, part]),
        }
        if previous is not None:
            row.update(
                temporal_id_change=(ids[:, part] != previous[0][:, part]).float().mean().item(),
                temporal_content_rmse=(values - previous[1][:, part]).square().mean().sqrt().item(),
            )
        groups.append(row)
    return {
        "groups": groups,
        "between_group_js_divergence": [[js_divergence(a, b) for b in probabilities] for a in probabilities],
        "between_group_content_cosine": [
            [mean_cosine(content[:, a], content[:, b]) for b in slices] for a in slices
        ],
        "content_distribution": distribution_stats(content),
        "content_excludes_positions": True,
    }


def intervene_ids(ids, donor, k_ref, mode, generator=None):
    changed = ids.clone()
    suffix = changed[:, k_ref:]
    if mode == "same_frame":
        suffix.copy_(ids.roll(ids.shape[1] // 2, dims=1)[:, k_ref:])
    elif mode == "other_frame":
        suffix.copy_(donor[:, k_ref:ids.shape[1]])
    elif mode == "shuffle":
        order = torch.randperm(suffix.shape[1], generator=generator).to(ids.device)
        suffix.copy_(ids[:, k_ref:][:, order])
    elif mode == "constant":
        suffix.zero_()  # Valid Code 0 is fixed across frames, never chosen using their content.
    elif mode == "all_constant":
        changed.zero_()
    elif mode != "original":
        raise ValueError("unknown diagnostic intervention")
    return changed


def rmse(a, b):
    return (a.float() - b.float()).square().mean().sqrt().item()


@torch.no_grad()
def decoder_interventions(model, ids, donor, context, original, budgets, metrics, generator):
    rows = []
    base_latent = model.decoder(model.vq.embed(ids[:, :model.config.k_ref]), context.latent_shape,
                                context.frame_type)
    base_image, _ = model.decode_latent(base_latent, context)
    for k in budgets:
        latent = model.decoder(model.vq.embed(ids[:, :k]), context.latent_shape, context.frame_type)
        normal, _ = model.decode_latent(latent, context)
        for mode in ("original", "same_frame", "other_frame", "shuffle", "constant", "all_constant"):
            changed = intervene_ids(ids[:, :k], donor, model.config.k_ref, mode, generator)
            altered_latent = model.decoder(model.vq.embed(changed), context.latent_shape, context.frame_type)
            image, _ = model.decode_latent(altered_latent, context)
            rows.append({
                "tokens": k, "intervention": mode, **metrics(original, image),
                "delta_latent_rmse": rmse(altered_latent, latent),
                "delta_image_rmse": rmse(image, normal),
                "changed_id_fraction": (changed != ids[:, :k]).float().mean().item(),
                "base_ids_preserved": torch.equal(changed[:, :model.config.k_ref], ids[:, :model.config.k_ref]),
                "diagnostic_only": True, "real_bitstream_rd": False,
            })
        rows.append({"tokens": k, "intervention": "base_only", **metrics(original, base_image),
                     "delta_latent_rmse": rmse(base_latent, latent),
                     "delta_image_rmse": rmse(base_image, normal), "diagnostic_only": True,
                     "real_bitstream_rd": False, "base_ids_preserved": True})
    return rows


@torch.no_grad()
def global_attention_components(projector, tokens, packet_tokens, include_positions=True):
    q = projector._heads(projector.q(projector.query.expand(tokens.shape[0], -1, -1)))
    positioned = tokens + projector.positions[:, :tokens.shape[1]] if include_positions else tokens
    k, v = projector.kv(projector.token_norm(positioned)).chunk(2, -1)
    k, v = projector._heads(k), projector._heads(v)
    dtype = projector.accumulation_dtype
    scores = q.to(dtype) @ k.to(dtype).transpose(-1, -2) / math.sqrt(q.shape[-1])
    weights = scores.softmax(-1)
    attended = (weights @ v.to(dtype)).transpose(1, 2).reshape(tokens.shape[0], -1, projector.dim)
    readout = projector.out(attended.to(projector.query.dtype))
    residual = projector.query + readout
    ff = projector.ff(projector.norm(residual))
    grid = (residual + ff).transpose(1, 2).reshape(tokens.shape[0], projector.dim, *projector.grid)
    by_grid = torch.stack([weights[..., s].sum(-1).mean((0, 1))
                          for s in group_slices(tokens.shape[1], packet_tokens)], -1)
    stats = {
        "query_norm": projector.query.norm().item(), "attention_readout_norm": readout.norm().item(),
        "attention_to_query_ratio": (readout.norm() / projector.query.norm().clamp_min(1e-20)).item(),
        "ffn_norm": ff.norm().item(), "grid_norm": grid.norm().item(),
        "group_attention_mass": by_grid.mean(0).cpu().tolist(),
        "per_grid_group_attention_mass": by_grid.cpu().tolist(),
    }
    return stats, grid, readout


@torch.no_grad()
def progressive_attention_components(projector, tokens, packet_tokens, include_positions=True):
    cache = projector.new_cache(tokens.shape[0])
    gates = []
    for part in group_slices(tokens.shape[1], packet_tokens):
        group = tokens[:, part]
        gates.append(projector.group_gate(group, part.start, include_positions).mean().item())
        delta = projector._group_delta(cache.query, group, part.start, include_positions)
        cache.accumulated = cache.accumulated + delta
    attended = cache.accumulated.transpose(1, 2).reshape(tokens.shape[0], -1, projector.dim)
    readout = projector.out(attended.to(projector.query.dtype))
    grid = projector._finish(cache.accumulated)
    stats = {"query_norm": projector.query.norm().item(), "attention_readout_norm": readout.norm().item(),
             "attention_to_query_ratio": (readout.norm() / projector.query.norm().clamp_min(1e-20)).item(),
             "grid_norm": grid.norm().item(), "group_attention_mass": gates,
             "per_grid_group_attention_mass": [gates] * (projector.grid[0] * projector.grid[1]),
             "attention_semantics": "independent packet softmax; group masses are gates, not global probabilities"}
    return stats, grid, readout


@torch.no_grad()
def attention_diagnostics(projector, embeddings, budgets, packet_tokens):
    rows, previous_grid, previous_readout, previous_k = [], None, None, 0
    components = progressive_attention_components if hasattr(projector, "group_gate") else global_attention_components
    for k in budgets:
        stats, grid, readout = components(projector, embeddings[:, :k], packet_tokens)
        _, no_position, _ = components(projector, embeddings[:, :k], packet_tokens, False)
        stats.update(tokens=k, position_grid_rmse=rmse(grid, no_position),
                     new_group_attention_mass=sum(stats["group_attention_mass"][previous_k // packet_tokens:]))
        if previous_grid is not None:
            stats.update(prefix_grid_rmse=rmse(grid, previous_grid),
                         prefix_attention_readout_rmse=rmse(readout, previous_readout))
        rows.append(stats)
        previous_grid, previous_readout, previous_k = grid, readout, k
    return rows


def gradient_groups(gradient, packet_tokens):
    squared = gradient.detach().float().square()
    total = squared.sum().clamp_min(1e-30)
    return [{"group": i, "l2": squared[:, part].sum().sqrt().item(),
             "rms": squared[:, part].mean().sqrt().item(),
             "squared_norm_share": (squared[:, part].sum() / total).item()}
            for i, part in enumerate(group_slices(gradient.shape[1], packet_tokens))]


def gradient_diagnostics(model, encoded, rgb, budgets):
    rows, stored = [], {}
    parameters = (encoded.vq.embeddings, encoded.features, model.tokenizer.queries)
    for k in budgets:
        rec, _ = model.decode_embeddings(encoded.vq.embeddings[:, :k], encoded.context)
        distortion = F.mse_loss(rec, rgb)
        derivatives = torch.autograd.grad(distortion, parameters, retain_graph=True)
        row = {"tokens": k, "mse": distortion.item()}
        for name, gradient in zip(("decoder_embedding", "vq_input", "query"), derivatives):
            row[name] = gradient_groups(gradient, model.config.packet_tokens)
        rows.append(row)
        stored[k] = derivatives[0].detach()
    conflicts = []
    for low, high in zip(budgets, budgets[1:]):
        groups = []
        for part in group_slices(low, model.config.packet_tokens):
            a, b = stored[low][:, part].flatten(), stored[high][:, part].flatten()
            groups.append(F.cosine_similarity(a[None], b[None]).item())
        conflicts.append({"low": low, "high": high, "shared_group_gradient_cosine": groups})
    return {"budgets": rows, "conflicts": conflicts, "source": "actual reconstruction MSE autograd"}


def diagnose_clip(model, clip, budgets, metric_names=("psnr", "lpips", "dists")):
    if model.vq.strategy == "continuous":
        raise ValueError("prefix diagnostics require a trained discrete VQ checkpoint")
    model.eval()
    metrics = MetricSuite(model.device, metric_names)
    state, previous = model.empty_state(), None
    groups, attention, gradients, interventions = [], [], [], []
    generator = torch.Generator().manual_seed(19)
    for t, frame in enumerate(clip):
        rgb = frame[None].to(model.device)
        encoded = model.encode_full(rgb, state)
        ids = encoded.vq.ids.detach()
        with torch.no_grad():
            content = model.vq.embed(ids)
            donor = model.encode_full(clip[(t + 1) % len(clip)][None].to(model.device), state).vq.ids
        common = {"frame": t, "frame_type": encoded.context.frame_type}
        stats = token_group_statistics(ids, content, model.decoder.projector.positions,
                                       model.config.packet_tokens, model.config.codebook_size, previous)
        stats.update(pre_vq_distribution=distribution_stats(encoded.features),
                     temporal_previous_frame_type=previous[2] if previous is not None else None,
                     temporal_comparison_same_frame_type=previous is not None and previous[2] == encoded.context.frame_type)
        groups.append(dict(common, **stats))
        attention.append(dict(common, prefixes=attention_diagnostics(
            model.decoder.projector, content, budgets, model.config.packet_tokens)))
        gradients.append(dict(common, **gradient_diagnostics(model, encoded, rgb, budgets)))
        rows = decoder_interventions(model, ids, donor, encoded.context, rgb, budgets, metrics, generator)
        interventions.extend(dict(common, **row) for row in rows)
        with torch.no_grad():
            _, feature = model.decode_ids(ids[:, :model.config.k_ref], encoded.context)
            state = encoded.context.commit(feature, ids_digest(ids[:, :model.config.k_ref])).detach()
        previous = ids, content, encoded.context.frame_type
        print(f"diagnosed frame {t} ({encoded.context.frame_type})", flush=True)
    return {"per_group_statistics": groups, "attention_mass": attention,
            "gradient_by_group": gradients, "token_interventions": interventions,
            "metric_errors": metrics.errors,
            "semantics": {"one_base_commit_per_frame": True, "fixed_history_per_intervention": True,
                          "fixed_constant_code_id": 0,
                          "donor_is_diagnostic_only": True, "interventions_are_valid_streams": False}}
