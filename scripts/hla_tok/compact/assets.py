"""Portable, content-addressed Compact projection, lift and scale assets."""

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from hla_tok.compression.profiles import Profile
from hla_tok.compression.scalar import StaticScalar
from hla_tok.compression.transform import CompactLift, CompactProjection, CompactTransform
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.streaming.compact import CompactCodec
from hla_tok.streaming.continuous import NATIVE_RANKS, RankOrder


def transform_for(stats: Mapping, kind: str, dimension: int) -> CompactTransform:
    if kind == "identity":
        if dimension != 32:
            raise ValueError("the identity transform requires 32 components")
        return CompactTransform()
    if kind not in ("global", "rank"):
        raise ValueError("unknown PCA transform kind")
    center = torch.tensor(stats[kind + "_mean"], dtype=torch.float32)
    basis = torch.tensor(stats[kind + "_basis"][..., :dimension], dtype=torch.float32)
    return CompactTransform.from_pca(center, basis)


def _transform_arrays(transform: CompactTransform) -> dict:
    projection, lift = transform.projection, transform.lift
    if projection is None or lift is None:
        return {}
    return {name: tensor.detach().cpu().numpy() for name, tensor in (
        ("center", projection.center), ("projection", projection.weight),
        ("lift_bias", lift.bias), ("lift", lift.weight),
    )}


def save_asset(path: Path, profile: Profile, kind: str, transform: CompactTransform,
               scalar: StaticScalar | None) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays = _transform_arrays(transform)
    if scalar is not None:
        arrays["scales"] = scalar.scales.detach().cpu().numpy()
    rank_count = transform.rank_count
    ranks = np.arange(rank_count, dtype=np.int16) if rank_count > 1 else np.array([-1], dtype=np.int16)
    temporary = path.with_suffix(".tmp.npz")
    np.savez(temporary, profile=np.array(profile.label), kind=np.array(kind), ranks=ranks,
             **{key: value.astype("<f4", copy=False) for key, value in arrays.items()})
    temporary.replace(path)
    return {"profile": profile.label, "projection": kind, "filename": path.name,
            "sha256": sha256_file(path), "storage_bytes": path.stat().st_size,
            "tensor_bytes": sum(value.nbytes for value in arrays.values()), "native_ranks": ranks.tolist()}


def _load_transform(archive: Mapping, profile: Profile, kind: str) -> CompactTransform:
    if kind == "identity":
        return CompactTransform()
    if kind not in ("global", "rank"):
        raise ValueError("unrecognized Compact transform")
    r, d = (len(NATIVE_RANKS) if kind == "rank" else 1), profile.dimension
    shapes = {"center": (r, 32), "projection": (r, 32, d), "lift_bias": (r, 32), "lift": (r, d, 32)}
    arrays = {name: archive[name] for name in shapes}
    if not all(value.shape == shapes[name] and value.dtype == np.dtype("<f4") and np.isfinite(value).all()
               for name, value in arrays.items()):
        raise ValueError("invalid affine Compact asset")
    t = {name: torch.from_numpy(value.astype("=f4", copy=False)) for name, value in arrays.items()}
    return CompactTransform(CompactProjection(t["center"], t["projection"]), CompactLift(t["lift_bias"], t["lift"]))


def load_asset(path: Path, spec: dict, profile: Profile, order: RankOrder, model_sha256: str,
               device: str = "cuda") -> tuple[CompactCodec, dict]:
    start = perf_counter()
    digest = sha256_file(path)
    if digest != spec["sha256"] or path.stat().st_size != spec["storage_bytes"]:
        raise ValueError("Compact Profile asset digest or length mismatch")
    with np.load(path, allow_pickle=False) as archive:
        profile_name, kind = archive["profile"].item(), archive["kind"].item()
        if (profile_name, kind) != (profile.label, spec["projection"]):
            raise ValueError("Compact Profile asset metadata differs")
        transform = _load_transform(archive, profile, kind)
        ranks = list(NATIVE_RANKS) if kind == "rank" else [-1]
        if archive["ranks"].tolist() != ranks:
            raise ValueError("Compact asset native Rank mapping differs")
        scalar = StaticScalar(torch.from_numpy(archive["scales"]), profile.bits) if profile.quantized else None
    transform.to(device).eval()
    if scalar is not None:
        scalar.to(device).eval()
    codec = CompactCodec(profile, transform, scalar, order, digest, model_sha256, device)
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()
    return codec, {"load_ms": 1000 * (perf_counter() - start),
                   "device_tensor_bytes": spec["tensor_bytes"] + codec.ranks.nbytes,
                   "wire_signature": codec.identity.signature.hex()}
