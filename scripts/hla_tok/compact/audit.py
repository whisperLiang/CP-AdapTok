"""Train-only C0 feature statistics and native RGB sensitivity."""

from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch import Tensor

from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.models.local import TOKEN_DIM
from hla_tok.streaming.compact import native_ranks
from hla_tok.streaming.continuous import (
    MAX_K,
    NATIVE,
    NATIVE_RANKS,
    ContinuousReceiver,
    RankOrder,
)
from scripts.hla_tok.compact.assets import transform_for
from scripts.hla_tok.compact.cache import TrainTokens, load_train_tokens
from scripts.hla_tok.compact.inputs import CONFIG, OUTPUT, TRAIN_PLAN, compact_inputs, compact_orders
from scripts.hla_tok.continuous.inputs import V3, Inputs, locked_output, run_arguments
from scripts.hla_tok.continuous.matrix import compute
from scripts.hla_tok.v2.protocol import json_at


def _moments(values: Tensor) -> tuple[np.ndarray, np.ndarray, int]:
    rows = values.reshape(-1, TOKEN_DIM).double()
    count = len(rows)
    center = rows.mean(0)
    shifted = rows - center
    return center.cpu().numpy(), (shifted.T @ shifted / max(1, count - 1)).cpu().numpy(), count


def _eigensystem(covariance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, basis = np.linalg.eigh(covariance)
    eigenvalues, basis = eigenvalues[..., ::-1].clip(0), basis[..., ::-1]
    pivots = np.take_along_axis(basis, np.abs(basis).argmax(axis=-2, keepdims=True), axis=-2)
    return eigenvalues, basis * np.sign(pivots)


def covariance(tokens: TrainTokens) -> dict:
    fit = tokens.fit_frames
    means, covs, counts = (np.asarray(items) for items in zip(
        *(_moments(tokens.rank(r)[:fit]) for r in NATIVE_RANKS), strict=True))
    total = counts.sum()
    center = np.average(means, axis=0, weights=counts)
    delta = means - center
    pooled = ((covs * (counts - 1)[:, None, None]).sum(0) + (delta.T * counts) @ delta) / (total - 1)
    eig, p = _eigensystem(pooled[None])
    rank_eig, rank_p = _eigensystem(covs)
    fallback = (counts < 33) | (rank_eig[:, -1] <= 1e-10 * rank_eig[:, 0])
    return {"global_mean": center[None], "global_covariance": pooled, "global_basis": p, "global_eigenvalues": eig,
            "rank_mean": np.where(fallback[:, None], center, means), "rank_samples": counts, "rank_empirical_mean": means,
            "rank_covariance": covs, "rank_basis": np.where(fallback[:, None, None], p, rank_p),
            "rank_eigenvalues": rank_eig, "native_rank_ids": np.asarray(NATIVE_RANKS),
            "rank_global_fallback": fallback}


def _spectrum_tables(output: Path, stats: dict, dimensions: list[int]) -> None:
    spectra: list[dict] = []
    ranks: list[dict] = []
    for scope, native, eigen in [("global", -1, stats["global_eigenvalues"][0])] + [
            ("rank", r, stats["rank_eigenvalues"][r]) for r in NATIVE_RANKS]:
        p = eigen / max(eigen.sum(), np.finfo(float).tiny)
        valid = p[p > 0]
        cumulative = np.cumsum(p)
        spectra.extend({"scope": scope, "native_rank": native, "component": i + 1,
                        "eigenvalue": eig, "explained_fraction": var, "cumulative_fraction": cum}
                       for i, (eig, var, cum) in enumerate(zip(eigen, p, cumulative, strict=True)))
        ranks.append({"scope": scope, "native_rank": native,
                      "samples": int(stats["rank_samples"].sum() if native < 0 else stats["rank_samples"][native]),
                      "effective_rank_entropy": float(np.exp(-(valid * np.log(valid)).sum())),
                      "participation_ratio": float(1 / max((p * p).sum(), np.finfo(float).tiny)),
                      "global_fallback": bool(native >= 0 and stats["rank_global_fallback"][native]),
                      "dimensions_at_95_percent": min(TOKEN_DIM, int(np.searchsorted(cumulative, .95) + 1)),
                      "dimensions_at_99_percent": min(TOKEN_DIM, int(np.searchsorted(cumulative, .99) + 1)),
                      **{f"variance_{d}d": cumulative[d - 1] for d in dimensions}})
    write_csv(output / "pca_spectrum.csv", spectra)
    write_csv(output / "rank_effective_dimension.csv", ranks)


def _rank_correlation(tokens: TrainTokens) -> np.ndarray:
    # The first 120 raster regions exist at every Rank.
    shared = tokens.tail.shape[1]
    fit = tokens.fit_frames
    values = torch.cat((tokens.full[:fit, :shared], tokens.tail[:fit]), dim=2)
    by_rank = values.permute(2, 0, 1, 3).reshape(len(NATIVE_RANKS), -1).float()
    standardized = torch.nn.functional.normalize(by_rank - by_rank.mean(dim=1, keepdim=True), dim=1)
    return (standardized @ standardized.T).cpu().numpy()


def _region_correlation(tokens: TrainTokens, *, axis: str) -> float:
    columns = NATIVE.columns
    step = 1 if axis == "horizontal" else columns
    result = torch.zeros(3, dtype=torch.float64, device=tokens.full.device)
    for block in (tokens.full[:tokens.fit_frames], tokens.tail[:tokens.fit_frames]):
        start = torch.arange(block.shape[1] - step, device=block.device)
        if axis == "horizontal":
            start = start[start % columns != columns - 1]
        values = block.float() - block.float().mean(dim=(0, 1), keepdim=True)
        a, b = values.index_select(1, start), values.index_select(1, start + step)
        result += torch.stack(((a * b).sum(dtype=torch.float64), a.square().sum(dtype=torch.float64),
                               b.square().sum(dtype=torch.float64)))
    dot, a_sq, b_sq = result.tolist()
    return dot / max((a_sq * b_sq) ** .5, 1e-30)


def _entropy(counts: np.ndarray) -> float:
    probabilities = counts[counts > 0] / counts.sum()
    return float(-(probabilities * np.log2(probabilities)).sum())


def _byte_distribution(values: Tensor) -> list[dict]:
    words = values.contiguous().view(torch.int16).cpu().numpy().astype("<i2", copy=False).view(np.uint8)
    columns = words.reshape(-1, 2)
    lo, hi = (np.bincount(columns[:, i], minlength=256) for i in (0, 1))
    return [{"byte_position": mode, "counts_0_to_255": counts.tolist(), "entropy_bits": _entropy(counts)}
            for mode, counts in (("all", lo + hi), ("low", lo), ("high", hi))]


def _distributions(native: Tensor, orders: dict, output: Path) -> None:
    groups = {"all": native}
    for name in ("O1", "O2"):
        ordered = orders[name].apply(native)
        groups[name + "_head_2040"] = ordered[:, :2040]
        groups[name + "_tail_2040"] = ordered[:, -2040:]
    records = []
    for segment, values in groups.items():
        vectors = values.flatten(0, 1).float()
        measures = torch.stack((vectors.mean(0), vectors.std(0), vectors.abs().mean(0),
                                vectors.min(0).values, vectors.max(0).values, vectors.abs().max(0).values))
        for channel, quantities in enumerate(measures.T.tolist()):
            records.append({"segment": segment, "channel": channel, "vectors": len(vectors),
                            **dict(zip(("mean", "std", "mean_abs", "min", "max", "max_abs"), quantities, strict=True))})
    write_csv(output / "token_distribution.csv", records)


def _sensitivity_image(receiver: ContinuousReceiver, values: Tensor, original: Tensor, order: RankOrder) -> float:
    return float((receiver.decode(values, order).float() - original).square().mean())


def _sensitivity(inputs: Inputs, config: dict, stats: dict, values: Tensor, output: Path) -> None:
    order = RankOrder.native("cuda")
    ranks = native_ranks(order, "cuda")
    receiver = ContinuousReceiver.from_model(inputs.model)
    holdout = json_at(TRAIN_PLAN)["holdout"][:config["calibration"]["sensitivity_frames"]]
    records, inverse = [], []
    for frame, vector in zip(holdout, values, strict=True):
        z = vector[None]
        source = transfer(inputs.reader.read("train", frame["video"], frame["frame"])[None])
        for kind in ("identity", "global", "rank"):
            transform = transform_for(stats, kind, TOKEN_DIM).cuda()
            for precision in ("fp32", "bf16") if kind != "identity" else ():
                with compute(precision):
                    estimate = transform.decode(transform.encode(z.float(), ranks), ranks)
                errors = (z.float() - estimate).abs()
                inverse.append(frame | {"projection": kind, "compute_dtype": precision,
                                        "rmse": float(errors.square().mean().sqrt()), "max_abs": float(errors.max())})
            with compute(config["compute_dtype"]):
                u = transform.encode(z, ranks)
                mse = _sensitivity_image(receiver, transform.decode(u, ranks), source, order)
                for channel in range(TOKEN_DIM):
                    erased = u.clone()
                    erased[..., channel] = 0
                    value = _sensitivity_image(receiver, transform.decode(erased, ranks), source, order)
                    records.append(frame | {"projection": kind, "channel": channel, "baseline_mse": mse,
                                            "ablated_mse": value, "rgb_mse_delta": value - mse})
    write_csv(output / "channel_sensitivity_per_frame.csv", records)
    summary = [{"projection": kind, "channel": i, "frames": len(holdout),
                "rgb_mse_delta": mean(r["rgb_mse_delta"] for r in records if (r["projection"], r["channel"]) == (kind, i))}
               for kind in ("identity", "global", "rank") for i in range(TOKEN_DIM)]
    write_csv(output / "channel_sensitivity.csv", summary)
    write_csv(output / "projection_reversibility.csv", inverse)


@torch.inference_mode()
def audit(inputs: Inputs, config: dict, output: Path) -> None:
    cache = load_train_tokens(inputs, config, output)
    print("Train token cache ready", flush=True)
    stats = covariance(cache)
    stats["rank_correlation"] = _rank_correlation(cache)
    np.savez(output / "rank_covariance.npz", **stats)
    print("Covariance and Rank spectra ready", flush=True)
    _spectrum_tables(output, stats, config["projection"]["dimensions"])
    native = cache.native()
    _distributions(native, compact_orders(), output)
    spatial = {axis: _region_correlation(cache, axis=axis) for axis in ("horizontal", "vertical")}
    count = config["calibration"]["sensitivity_frames"]
    _sensitivity(inputs, config, stats, native[cache.fit_frames:cache.fit_frames + count], output)
    write_json(output / "representation_audit.json", {
        "status": "PASS", "train_frames": len(native), "fit_frames": cache.fit_frames,
        "holdout_frames": len(native) - cache.fit_frames, "native_shape": config["native_shape"],
        "tokens_per_frame": MAX_K, "feature_shapes": [list(cache.full.shape), list(cache.tail.shape)],
        "plan_sha256": sha256_file(TRAIN_PLAN), "protocol_sha256": sha256_file(CONFIG),
        "covariance_sha256": sha256_file(output / "rank_covariance.npz"), "adjacent_region_correlation": spatial,
        "byte_histograms": _byte_distribution(native), "rank_correlation_shared_regions": cache.tail.shape[1],
        "pca_fallback_ranks": np.flatnonzero(stats["rank_global_fallback"]).tolist(),
        "pixel_reads": dict(inputs.reader.counts), "test_pixel_reads": inputs.reader.counts["test"],
    })
    inputs.finish(output / "audit_inputs.json")


def main() -> None:
    parser = run_arguments(__doc__)
    parser.set_defaults(output=OUTPUT, v3=V3)
    args = parser.parse_args()
    with locked_output(args.output):
        inputs, config = compact_inputs(args.v3, args.output)
        audit(inputs, config, args.output)


if __name__ == "__main__":
    main()
