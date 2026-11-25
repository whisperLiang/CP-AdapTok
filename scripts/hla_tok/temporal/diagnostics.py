"""Paired native temporal diagnostics after the frozen baseline replay."""

from hashlib import sha256
from pathlib import Path

import numpy as np
import torch
from torch.nn.functional import adaptive_avg_pool2d

from hla_tok.compression.profiles import PROFILES_BY_LABEL
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.streaming.compact_wire import deserialize
from hla_tok.streaming.continuous import NATIVE, encode_continuous
from hla_tok.temporal.layout import LAYOUT
from hla_tok.temporal.profile import RANK_COMPONENT_SHAPE
from hla_tok.temporal.quantization import quantize, target_lattice
from scripts.hla_tok.compact.assets import load_asset
from scripts.hla_tok.continuous.inputs import open_inputs, run_arguments
from scripts.hla_tok.continuous.matrix import compute, groups
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.temporal.engine import Link, prediction_coordinates, rms
from scripts.hla_tok.temporal.inputs import COMPACT, RESULTS, Assets, Setting, source_frame, validation_plan
from scripts.hla_tok.v2.protocol import json_at

MODES = ("A0 absolute", "A1 oracle", "A2 CP decoded", "A3 copy CLOSED", "A4 linear CLOSED")
CODES = 16
STARTS = np.array([LAYOUT.tokens_for_rank(r)[0] for r in range(RANK_COMPONENT_SHAPE[0])])


def _reduce(values: np.ndarray) -> np.ndarray:
    return np.add.reduceat(values, STARTS, axis=0)


def _regions(rgb: torch.Tensor, previous: torch.Tensor) -> np.ndarray:
    motion = (rgb - previous).abs().mean(dim=1, keepdim=True)
    return adaptive_avg_pool2d(NATIVE.pad(motion), LAYOUT.grid).flatten().cpu().numpy()[LAYOUT.regions]


class Moments:
    def __init__(self):
        self.size = np.diff(np.append(STARTS, LAYOUT.token_count)) * LAYOUT.components
        self.samples = 0
        self.moments = np.zeros((5, len(STARTS)))
        self.squared = np.zeros(RANK_COMPONENT_SHAPE)
        self.signal = np.zeros(RANK_COMPONENT_SHAPE)
        self.zero = np.zeros(RANK_COMPONENT_SHAPE)
        self.near = np.zeros(RANK_COMPONENT_SHAPE)
        self.hist = np.zeros((len(STARTS), CODES), dtype=np.int64)
        take = [LAYOUT.tokens_for_rank(r)[::8] for r in range(len(STARTS))]
        self.samples_by_rank = take
        self.cov_samples: list[np.ndarray] = []
        self.sample_slots = np.concatenate(take)

    def update(self, source: np.ndarray, predicted: np.ndarray, scale: np.ndarray, step: np.ndarray) -> None:
        residual: np.ndarray = source - predicted
        self.samples += 1
        for i, values in enumerate((source, predicted, source * predicted, source**2, predicted**2)):
            self.moments[i] += _reduce(values.sum(axis=1))
        self.squared += _reduce(residual**2)
        self.signal += _reduce(source**2)
        self.zero += _reduce(residual == 0)
        self.near += _reduce(np.abs(residual) <= (scale * step * 0.5))
        symbols, _ = quantize(target_lattice(source, scale), target_lattice(predicted, scale), step, 4)
        bins = LAYOUT.ranks[:, None] * CODES + symbols + 8
        self.hist += np.bincount(bins.ravel(), minlength=len(STARTS) * CODES).reshape(self.hist.shape)
        self.cov_samples.append(residual[self.sample_slots].astype(np.float32))

    def corr(self, mode: str) -> list[dict]:
        x, y, xy, xx, yy = self.moments
        n = self.size * self.samples
        spread = np.sqrt(np.maximum((xx - x**2 / n) * (yy - y**2 / n), 0))
        return [{"method": mode, "rank": r, "pearson": float((xy[r] - x[r] * y[r] / n[r]) / spread[r])
                 if spread[r] else None, "pairs": self.samples, "observations": int(n[r])}
                for r in range(len(STARTS))]

    def by_component(self, mode: str) -> list[dict]:
        counts = self.size / LAYOUT.components * self.samples
        per_rank_hist = self.hist / self.hist.sum(axis=1, keepdims=True)
        h = -(per_rank_hist * np.log2(np.maximum(per_rank_hist, 1e-30))).sum(axis=1)
        return [{"method": mode, "rank": r, "component": c, "mse": self.squared[r, c] / counts[r],
                 "absolute_energy": self.signal[r, c] / counts[r],
                 "residual_to_absolute_energy": self.squared[r, c] / max(self.signal[r, c], 1e-30),
                 "zero_fraction": self.zero[r, c] / counts[r], "near_zero_fraction": self.near[r, c] / counts[r],
                 "rank_int4_entropy_bits_per_symbol": h[r]} for r, c in np.ndindex(*RANK_COMPONENT_SHAPE)]

    def spectrum(self, mode: str) -> list[dict]:
        matrix = np.stack(self.cov_samples)
        output: list[dict] = []
        pointer = 0
        for r, slots in enumerate(self.samples_by_rank):
            data = matrix[:, pointer:pointer + len(slots)].reshape(-1, LAYOUT.components)
            covariance = np.cov(data, rowvar=False)
            eigen = np.maximum(np.linalg.eigvalsh(covariance)[::-1], 0)
            p = eigen / max(eigen.sum(), 1e-30)
            n = np.exp(-(p * np.log(np.maximum(p, 1e-30))).sum())
            pr = 1 / max(np.square(p).sum(), 1e-30)
            output.extend({"method": mode, "rank": r, "eigen_index": i, "covariance_eigenvalue": value,
                           "effective_dimension": n, "participation_ratio": pr, "trace": eigen.sum()}
                          for i, value in enumerate(eigen))
            pointer += len(slots)
        return output


def _previous_cp(cp, cp_order, native, transform, frame: int) -> np.ndarray:
    encoded = cp.frame(cp.encode_tokens(native, LAYOUT.token_count), frame).serialize_range(0, LAYOUT.token_count)
    raw = deserialize(encoded)
    if raw.identity != cp.identity:
        raise RuntimeError("diagnostic CP binding differs")
    restored = cp.lift(cp.dequantize(cp.unpack(raw.payload).cuda()))
    z = cp_order.restore_full(restored)
    return transform.project(z)[0].cpu().numpy()


@torch.inference_mode()
def run(v3: Path, output: Path) -> None:
    inputs = open_inputs(v3)
    assets = Assets(output)
    synth = assets.synthesis(inputs.model)
    scale = assets.arrays["scale"][LAYOUT.ranks]
    step = assets.arrays["i4"][LAYOUT.ranks]
    fixed = Setting("T0 fixed", "B2", "S0", "copy")
    pcopy = assets.profile(fixed)
    plinear = pcopy.with_predictor("linear", refresh=32)
    chart = NativeQuality()
    spec = json_at(COMPACT / "compact_profile_manifest.json")["Q16-4_rank"]
    cp, _ = load_asset(COMPACT / "profiles" / spec["filename"], spec, PROFILES_BY_LABEL["Q16-4"],
                       assets.order, inputs.checkpoint["state_dict_sha256"])
    support_all = np.ones(LAYOUT.token_count, dtype=bool)
    statistics = {mode: Moments() for mode in MODES}
    quality_rows: list[dict] = []
    motion_rows: list[dict] = []
    fingerprints: list[dict] = []
    for clip in validation_plan(inputs)[0]:
        systems = (Link(pcopy), Link(plinear))
        old = None
        for i in range(clip.length):
            rgb = source_frame(inputs.reader, clip, i)
            with compute("bf16"):
                native = encode_continuous(inputs.model, rgb)
                cp_current = _previous_cp(cp, assets.order, native, assets.transform, i)
            source = assets.transform.project(native)[0].cpu().numpy()
            target = target_lattice(source, scale)
            delivered = [link.send(target, i, profile.max_base_bytes, payload_cap=None, channel="base-only")
                         for link, profile in zip(systems, (pcopy, plinear), strict=True)]
            copied, linear = (prediction_coordinates(out, profile) * scale
                              for out, profile in zip(delivered, (pcopy, plinear), strict=True))
            first, last = delivered
            assert first.receiver is not None and last.receiver is not None
            fingerprints.append({"video": clip.video, "frame": i, "copy": first.receiver.digest.hex(),
                                 "linear": last.receiver.digest.hex(), "copy_type": first.frame_type,
                                 "support_digest": sha256(first.receiver.support.tobytes()).hexdigest()})
            if old is not None:
                previous, cp_previous, prev_rgb = old
                motion = _regions(rgb, prev_rgb)
                chart.prepare(rgb)
                predictions = (np.zeros_like(source), previous, cp_previous, copied, linear)
                masks = (support_all, support_all, support_all, first.receiver.support, last.receiver.support)
                for method, value, support in zip(MODES, predictions, masks, strict=True):
                    stats = statistics[method]
                    stats.update(source, value, scale, step)
                    latent = torch.from_numpy(value).cuda()[None]
                    with compute("bf16"):
                        image = synth.render_coefficients(latent, support)
                    residual = source - value
                    squared = np.square(residual).mean(axis=1)
                    sample = {"video": clip.video, "frame": i, "method": method, "motion_rgb_l1": float(motion.mean()),
                              "normalized_scene_threshold_trigger": bool(motion.mean() > 0.18),
                              "base_refresh": first.frame_type == "I", "latent_prediction_rms": rms(value),
                              "latent_residual_rms": rms(residual)}
                    quality_rows.append(sample | chart.score(image))
                    strata: tuple[tuple[str, np.ndarray], ...] = (("low", motion <= 2 / 255), ("middle", (motion > 2 / 255) & (motion <= 8 / 255)),
                              ("high", motion > 8 / 255))
                    for label, selection in strata:
                        if selection.any():
                            motion_rows.append(sample | {"motion_stratum": label, "tokens": int(selection.sum()),
                                                         "residual_mse": float(squared[selection].mean())})
            old = source, cp_current, rgb
        print(f"Temporal diagnostic {clip.video}: {clip.length} native frames", flush=True)
    write_csv(output / "rank_temporal_correlation.csv", [r for m, a in statistics.items() for r in a.corr(m)])
    write_csv(output / "residual_statistics.csv", [r for m, a in statistics.items() for r in a.by_component(m)])
    write_csv(output / "residual_spectrum.csv", [r for m, a in statistics.items() for r in a.spectrum(m)])
    write_csv(output / "residual_histogram.csv", [{"method": m, "rank": r, "int4_code": c - 8, "count": int(value)}
              for m, a in statistics.items() for r, series in enumerate(a.hist) for c, value in enumerate(series)])
    write_csv(output / "prediction_only_rd.csv", [tags | summarize(items) | {"history": "CLOSED" if "CLOSED"
              in tags["method"] else "FIXED diagnostic"} for tags, items in groups(quality_rows, ("method",))])
    write_csv(output / "prediction_only_per_frame.csv", quality_rows)
    write_csv(output / "motion_stratified_analysis.csv", motion_rows)
    write_json(output / "diagnostic_reference_audit.json", {"status": "PASS", "frames": fingerprints})
    inputs.finish(output / "diagnostic_inputs.json")


if __name__ == "__main__":
    parser = run_arguments(__doc__)
    parser.set_defaults(output=RESULTS)
    args = parser.parse_args()
    run(args.v3, args.output)
