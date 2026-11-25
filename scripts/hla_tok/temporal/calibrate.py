"""Fit the lattice, I/P steps, and static allocation using contiguous Train clips."""

from pathlib import Path

import numpy as np
import torch

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.streaming.temporal import encode_frame
from hla_tok.streaming.temporal_wire import NO_DIGEST, record_bytes
from hla_tok.temporal.allocation import base_scan
from hla_tok.temporal.layout import LAYOUT
from hla_tok.temporal.profile import RANK_COMPONENT_SHAPE, Profile
from hla_tok.temporal.quantization import quantize, reconstruct, target_lattice
from hla_tok.temporal.sparse import pack_block
from hla_tok.temporal.transform import TemporalSynthesis
from scripts.hla_tok.compact.inputs import compact_orders
from scripts.hla_tok.continuous.inputs import Inputs, open_inputs, run_arguments
from scripts.hla_tok.continuous.matrix import compute
from scripts.hla_tok.temporal.inputs import (
    COMPACT,
    CONFIG,
    RESULTS,
    Clip,
    freeze_clips,
    frozen_transform,
    project_source,
    source_frame,
)
from scripts.hla_tok.v2.protocol import json_at


def _step_candidates(values: np.ndarray, bits: int, percentiles: list[float]) -> np.ndarray:
    peak = np.percentile(np.abs(values), percentiles, axis=0)
    return np.maximum(1, np.ceil(peak / (2**(bits - 1) - 1))).astype(np.int32)


def _best(candidates: np.ndarray, errors: list[np.ndarray]) -> np.ndarray:
    # The preset order prefers the wider range when held-out MSE ties.
    index = len(errors) - 1 - np.argmin(np.stack(errors[::-1]), axis=0)
    return np.take_along_axis(candidates, index[None], axis=0)[0]


def _tune_rank(fit: list[np.ndarray], holdout: list[np.ndarray], scale: np.ndarray,
               bits: int, percentiles: list[float]) -> tuple[np.ndarray, np.ndarray]:
    q_fit, q_held = ([target_lattice(arr, scale) for arr in group] for group in (fit, holdout))
    intra = np.concatenate(q_fit).reshape(-1, LAYOUT.components)
    innovation = np.concatenate([np.diff(q.astype(np.int32), axis=0) for q in q_fit]).reshape(-1, LAYOUT.components)
    candidates_i, candidates_p = (_step_candidates(values, bits, percentiles) for values in (intra, innovation))
    joined = np.concatenate(q_held)
    errors_i = []
    for step in candidates_i:
        codes, _ = quantize(joined, 0, step, bits)
        errors_i.append(np.square(reconstruct(0, codes, step).astype(np.float32) - joined).mean(axis=(0, 1)))
    step_i = _best(candidates_i, errors_i)
    errors_p = []
    for step in candidates_p:
        error = np.zeros(LAYOUT.components, dtype=np.float64)
        for clip in q_held:
            start, _ = quantize(clip[0], 0, step_i, bits)
            history = reconstruct(0, start, step_i)
            for target in clip[1:]:
                codes, _ = quantize(target, history, step, bits)
                history = reconstruct(history, codes, step)
                error += np.square(history.astype(np.float32) - target).sum(axis=0)
        errors_p.append(error)
    return step_i, _best(candidates_p, errors_p)


def lattice_fit(fit: list[np.ndarray], holdout: list[np.ndarray], config: dict) -> dict:
    knobs = config["scales"]
    arrays = {k: np.empty(RANK_COMPONENT_SHAPE, dtype=np.float32)
              for k in ("scale", "S1_absolute", "S1_temporal")}
    arrays.update({f"{mode}{bits}": np.empty(RANK_COMPONENT_SHAPE, dtype=np.int32)
                   for mode in "ip" for bits in config["quantizer_bits"]})
    for rank in range(RANK_COMPONENT_SHAPE[0]):
        slots = LAYOUT.tokens_for_rank(rank)
        fitted = [clip[:, slots[::8]] for clip in fit]
        held = [clip[:, slots[::8]] for clip in holdout]
        peak = np.maximum.reduce([np.abs(clip[:, slots[0]:slots[-1] + 1]).max(axis=(0, 1)) for clip in fit])
        scale = np.maximum(peak / knobs["lattice_train_peak_steps"], 1e-7)
        arrays["scale"][rank] = scale
        arrays["S1_absolute"][rank] = np.square(np.concatenate(fitted)).mean(axis=(0, 1))
        diffs = np.concatenate([np.diff(clip, axis=0) for clip in fitted])
        arrays["S1_temporal"][rank] = np.square(diffs).mean(axis=(0, 1))
        for bits in config["quantizer_bits"]:
            step_i, step_p = _tune_rank(fitted, held, scale, bits, knobs["clip_percentiles"])
            arrays[f"i{bits}"][rank], arrays[f"p{bits}"][rank] = step_i, step_p
    return arrays


def _pilot_profile(arrays: dict, config: dict) -> Profile:
    order = compact_orders("cpu")["O2"]
    knobs = config["base_profiles"]["B2"]
    base = base_scan(order, knobs["cap"], knobs["ranks"], knobs["components"], 4, 256)
    return Profile(2, base, len(base), arrays["i4"], arrays["p4"], NO_DIGEST, base_cap=knobs["cap"])


def _prediction(clip: np.ndarray, index: int, scale: np.ndarray, profile: Profile) -> np.ndarray:
    base = None
    for t, frame in enumerate(clip[:index]):
        target = target_lattice(frame, scale)
        base = encode_frame(target, t, base, profile, profile.max_base_bytes).base
    if base is None:
        raise ValueError("RGB allocation must use an established Train base")
    dense = np.zeros_like(clip[index], dtype=np.int16)
    dense.reshape(-1)[profile.scan] = base.coordinates[profile.scan_base_order]
    return dense


def _rank_sum(values: np.ndarray) -> np.ndarray:
    starts = [int(LAYOUT.tokens_for_rank(r)[0]) for r in range(RANK_COMPONENT_SHAPE[0])]
    return np.add.reduceat(values, starts, axis=0)


def _rgb_benefit(inputs: Inputs, synth: TemporalSynthesis, selected: list[Clip], fit: list[np.ndarray],
                 arrays: dict, profile: Profile) -> dict:
    shape = (LAYOUT.token_count, LAYOUT.components)
    scale = arrays["scale"][LAYOUT.ranks]
    support = np.ones(LAYOUT.token_count, dtype=bool)
    totals = {name: np.zeros(RANK_COMPONENT_SHAPE, dtype=np.float64) for name in ("absolute", "temporal")}
    for clip, projected in zip(selected[:4], fit[:4], strict=True):
        for mode, offset in (("absolute", 0), ("temporal", 16)):
            pred = np.zeros(shape, dtype=np.int16) if offset == 0 else _prediction(projected, offset, scale, profile)
            start = pred * scale
            rgb = source_frame(inputs.reader, clip, offset)
            point = torch.tensor(start, device="cuda", requires_grad=True)[None]
            with compute("bf16"):
                image = synth.render_coefficients(point, support)
            grad, = torch.autograd.grad((image.float() - rgb).square().mean(), point)
            bound_gain = -grad[0].float().cpu().numpy() * (projected[offset] - start)
            totals[mode] += np.maximum(_rank_sum(bound_gain), 0)
    return {name: values / 4 for name, values in totals.items()}


def _actual_bits(holdout: list[np.ndarray], arrays: dict, profile: Profile) -> dict:
    scale = arrays["scale"][LAYOUT.ranks]
    steps_i, steps_p = (arrays[k][LAYOUT.ranks] for k in ("i4", "p4"))
    inter = steps_i.copy()
    inter.reshape(-1)[profile.scan] = steps_p.flat[profile.scan]
    totals = {name: np.zeros(RANK_COMPONENT_SHAPE) for name in ("absolute", "temporal")}
    for clip in holdout:
        for mode, offset in (("absolute", 0), ("temporal", 16)):
            pred = 0 if offset == 0 else _prediction(clip, offset, scale, profile)
            steps = steps_i if offset == 0 else inter
            code, _ = quantize(target_lattice(clip[offset], scale), pred, steps, 4)
            for rank in range(RANK_COMPONENT_SHAPE[0]):
                tokens = LAYOUT.tokens_for_rank(rank)
                for component, words in enumerate(code[tokens[0]:tokens[-1] + 1].T):
                    block = pack_block(words, 4, True)
                    totals[mode][rank, component] += 8 * len(record_bytes(
                        0, 0, words.size, False, block.mode, block.payload))
    return {name: values / len(holdout) for name, values in totals.items()}


def _read_clips(inputs: Inputs, transform, selected: list[Clip]) -> list[np.ndarray]:
    values = []
    for clip in selected:
        values.append(np.stack([project_source(inputs, transform, source_frame(inputs.reader, clip, offset))
                                for offset in range(clip.length)]))
        print(f"Train clip {clip.video} {clip.start}–{clip.start + clip.length - 1}", flush=True)
    return values


def run(v3: Path, output: Path) -> None:
    if json_at(output / "baseline_reproduction.json")["status"] != "PASS":
        raise RuntimeError("baseline reproduction must pass before temporal calibration")
    config = json_at(CONFIG)
    inputs = open_inputs(v3)
    train = freeze_clips(inputs, output, config)
    transform, pca = frozen_transform()
    fit, holdout = (_read_clips(inputs, transform, train[name]) for name in ("fit", "holdout"))
    arrays = lattice_fit(fit, holdout, config)
    pilot = _pilot_profile(arrays, config)
    scale = torch.as_tensor(arrays["scale"], device="cuda")
    synth = TemporalSynthesis.from_model(inputs.model, transform, scale)
    gain = _rgb_benefit(inputs, synth, train["fit"], fit, arrays, pilot)
    bits = _actual_bits(holdout, arrays, pilot)
    for name in gain:
        arrays[f"S2_{name}"] = (gain[name] / bits[name]).astype(np.float32)
    path = output / "lattice.npz"
    temporary = output / "lattice.tmp.npz"
    np.savez(temporary, **arrays)
    temporary.replace(path)
    order = COMPACT.parent / "continuous/rank_order_manifest.json"
    write_json(output / "temporal_assets.json", {
        "lattice_sha256": sha256_file(path), "pca_sha256": pca, "rank_order_sha256": sha256_file(order),
        "model_sha256": inputs.checkpoint["state_dict_sha256"], "checkpoint_sha256": inputs.checkpoint["sha256"],
        "protocol_sha256": sha256_file(CONFIG), "sequences_sha256": sha256_file(output / "sequence_manifest.json"),
        "lattice_bytes": path.stat().st_size, "reconstruction_rounding": "nearest ties to even, int16 checked",
        "sampled_regions_for_steps": "every eighth actual native region", "rgb_sensitivity_clips": 4,
    })
    records = [{"rank": r, "component": c, **{name: float(values[r, c]) for name, values in arrays.items()},
                **{f"holdout_{name}_physical_bits": float(values[r, c]) for name, values in bits.items()}}
               for r, c in np.ndindex(*RANK_COMPONENT_SHAPE)]
    write_csv(output / "scan_calibration.csv", records)
    inputs.finish(output / "calibration_inputs.json")


if __name__ == "__main__":
    parser = run_arguments(__doc__)
    parser.set_defaults(output=RESULTS)
    options = parser.parse_args()
    run(options.v3, options.output)
