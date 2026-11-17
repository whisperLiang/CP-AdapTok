"""Measured image/latent/token errors and explicit aggregation definitions."""

import math

import torch
from torch.nn import functional as F


def psnr(mse):
    return -10 * math.log10(max(mse, 1e-12))


def distortion(target, actual):
    mse = F.mse_loss(target.float(), actual.float()).item()
    energy = target.float().square().mean().item()
    return {'latent_mse': mse, 'latent_relative_rmse': math.sqrt(mse / max(energy, 1e-12))}


def token_statistics(features, vq=None, ids=None):
    features = features.detach().float()
    stats = {'continuous_variance': features.var(dim=1, unbiased=False).mean().item(),
             'total_feature_variance': features.var(unbiased=False).item(),
             'group_variance': [group.var(dim=1, unbiased=False).mean().item()
                                for group in features.split(128, dim=1)]}
    if ids is not None:
        z = vq.project_features(features)
        quantized = F.embedding(ids, vq.codebook_vectors())
        counts = torch.bincount(ids.flatten(), minlength=vq.codebook.num_embeddings).float()
        probabilities = counts / counts.sum()
        stats.update(codebook_used=int((counts > 0).sum()),
                     codebook_usage=(counts > 0).float().mean().item(),
                     perplexity=(-(probabilities * probabilities.clamp_min(1e-12).log()).sum()).exp().item(),
                     token_quantization_mse=F.mse_loss(z, quantized).item())
    return stats


def content_variants(features, other):
    permutation = torch.randperm(features.shape[1], generator=torch.Generator().manual_seed(20261117))
    return {'normal': features, 'constant': features.mean(1, keepdim=True).expand_as(features),
            'shuffled': features[:, permutation.to(features.device)], 'other_frame': other}


@torch.no_grad()
def content_sensitivity(model, encoded, other_features, path='R1'):
    """Nonstandard diagnostics only: fixed decoder positions and frozen history."""
    variants = content_variants(model.tokens(encoded), other_features)
    outputs = {}
    for name, features in variants.items():
        embeddings, _, _ = model._embeddings(features, path)
        latent = model.decoder(embeddings, encoded.context.latent_shape, encoded.context.frame_type)
        rgb, _ = model.decode_latent(latent, encoded.context)
        outputs[name] = (latent, rgb)
    normal = outputs['normal']
    return {name: {'latent_delta_mse': F.mse_loss(out[0].float(), normal[0].float()).item(),
                   'rgb_delta_mse': F.mse_loss(out[1].float(), normal[1].float()).item()}
            for name, out in outputs.items()}


def _means(rows):
    mse = sum(r['mse'] * r['pixels'] for r in rows) / sum(r['pixels'] for r in rows)
    result = {'frames': len(rows), 'global_mse': mse, 'global_mse_psnr': psnr(mse),
              'mean_frame_psnr': sum(r['psnr'] for r in rows) / len(rows)}
    for name in ('lpips', 'dists', 'ms-ssim', 'latent_mse', 'latent_relative_rmse'):
        values = [r[name] for r in rows if r.get(name) is not None]
        result[name] = sum(values) / len(values) if values else None
    return result


def aggregate(rows):
    if not rows:
        raise ValueError('cannot aggregate empty frame measurements')
    result = _means(rows)
    result['videos'] = {video: _means([r for r in rows if r['video'] == video])
                        for video in sorted({r['video'] for r in rows})}
    result['mean_video_psnr'] = sum(v['mean_frame_psnr'] for v in result['videos'].values()) / len(result['videos'])
    for frame_type in ('I', 'P'):
        selected = [r for r in rows if r['frame_type'] == frame_type]
        result[frame_type] = _means(selected) if selected else None
    return result


def module_norms(model, gradients=True):
    modules = {'tokenizer': model.tokenizer, 'project_in': model.vq.project_in,
               'project_out': model.vq.project_out, 'decoder': model.decoder, 'codebook': model.vq.codebook}
    result = {}
    for name, module in modules.items():
        tensors = [p.grad if gradients else p for p in module.parameters()]
        result[name] = math.sqrt(sum(t.detach().float().square().sum().item() for t in tensors if t is not None))
    return result
