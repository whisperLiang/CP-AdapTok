"""Rate-aware Compact adapter objective with frozen C0 synthesis."""

from random import Random

import torch
from torch import Tensor
from torch.nn import functional as F

from hla_tok import HLATok
from hla_tok.compression.profiles import Profile
from hla_tok.compression.scalar import StaticScalar
from hla_tok.compression.transform import CompactTransform
from hla_tok.models.codec import recover_grid, recover_groups
from hla_tok.streaming.compact import CompactCodec
from hla_tok.streaming.compact_wire import max_budget
from hla_tok.streaming.continuous import MAX_K, NATIVE, Order, RankOrder
from hla_tok.training.losses import ReconstructionLoss
from scripts.hla_tok.continuous.inputs import verify_model_state


def budget_choices(profile: Profile, caps: list[int]) -> tuple[int, ...]:
    return tuple(sorted({max_budget(cap, profile) for cap in caps}))


def adapter_latents(transform: CompactTransform, scalar: StaticScalar, values: Tensor,
                    ranks: Tensor, *, soft: bool) -> Tensor:
    if not soft:
        with torch.no_grad():
            u = transform.encode(values, ranks)
            return scalar.decode(scalar.encode(u, ranks), ranks)
    u = transform.encode(values, ranks).float()
    scale = scalar.scale(u, ranks)
    noisy = u + (torch.rand_like(u) - 0.5) * scale
    return torch.clamp(noisy, min=scalar.low * scale, max=scalar.high * scale)


def recover(model: HLATok, values: Tensor, order: RankOrder) -> Tensor:
    if values.shape[1] == MAX_K:
        return recover_grid(model.local_decoder, order.restore_full(values), NATIVE)
    if order.name == Order.O0:
        return recover_grid(model.local_decoder, values, NATIVE)
    return recover_groups(model.local_decoder, order.recovery_groups(values), NATIVE)


def _loss(model: HLATok, loss: ReconstructionLoss, source: Tensor, values: Tensor, order: RankOrder, crops: tuple):
    reconstructed = model.rgb_decoder(recover(model, values, order), NATIVE)
    terms = loss(source, reconstructed, crops)
    return sum(loss.weights[key] * value for key, value in terms.items())


def loss_for_budget(model: HLATok, codec: CompactCodec, loss: ReconstructionLoss, source: Tensor, native: Tensor,
                    rms: Tensor, budget: int, crops: tuple, *, soft: bool, latent_weight: float) -> tuple[Tensor, Tensor]:
    ordered = codec.order.apply(native)
    scalar = codec.scalar
    if scalar is None:
        raise ValueError("adaptation requires frozen scalar scales")
    u = adapter_latents(codec.transform, scalar, ordered, codec.ranks, soft=soft)
    z = codec.lift(u)
    whole = _loss(model, loss, source, z, codec.order, crops)
    small = whole if budget == MAX_K else _loss(model, loss, source, z[:, :budget], codec.order, crops)
    latent = F.mse_loss(z.float() / rms, ordered.float() / rms)
    return (whole + small) * 0.5 + latent_weight * latent, latent


def crops_for_step(seed: int, cursor: int) -> tuple[tuple[int, int], ...]:
    rng = Random(seed + cursor * 128)
    return tuple((rng.randrange(NATIVE.height - 255), rng.randrange(NATIVE.width - 255)) for _ in range(2))


def assert_frozen(model: torch.nn.Module, expected_sha256: str) -> str:
    if any(p.requires_grad or p.grad is not None for p in model.parameters()):
        raise RuntimeError("trainable or updated C0 parameters detected")
    return verify_model_state(model, expected_sha256)
