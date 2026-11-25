"""Matched Validation rate–distortion and independent closed histories."""

from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from hashlib import sha256
from pathlib import Path

import numpy as np
import torch

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.evaluation.reconstruction import save_native
from hla_tok.streaming.continuous_wire import wire_rate
from hla_tok.temporal.layout import LAYOUT
from hla_tok.temporal.profile import RANK_COMPONENT_SHAPE
from hla_tok.temporal.quantization import target_lattice
from scripts.hla_tok.continuous.inputs import open_inputs, run_arguments
from scripts.hla_tok.continuous.matrix import compute, groups
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.temporal.engine import Delivered, Link, prediction_coordinates, rms, temporal_error
from scripts.hla_tok.temporal.inputs import (
    CONFIG,
    RESULTS,
    Assets,
    Setting,
    project_source,
    source_frame,
    validation_plan,
)
from scripts.hla_tok.v2.protocol import ROOT, json_at

FIXED = Setting("T0 fixed", "B2", "S0", "copy")
MODES = ("global_psnr", "lpips", "dists")


@dataclass(frozen=True)
class Cap:
    bandwidth: str
    total: int
    payload: int | None = None
    basis: str = "total_wire"


@dataclass(frozen=True)
class Arm:
    setting: Setting
    cap: Cap


def experiments(config):
    caps = {name: Cap(name, size) for name, size in config["wire_caps"].items()}
    common = list(caps.values()) + [
        Cap("C1 envelope", config["equal_c1_wire_bytes"], basis="total_wire_f13w"),
        Cap("C1 payload", 1_000_000, config["equal_payload_bytes"], "payload_only"),
    ]
    practical = [caps[str(rate)] for rate in (5, 10, 20, 40)]
    low = practical[:2]
    intra = replace(FIXED, label="Intra fixed", predictor="absolute")
    rank = replace(FIXED, label="T0-RA", scan="S1", score_basis="absolute")
    main = [intra, FIXED, replace(FIXED, label="T0 linear", predictor="linear"),
            replace(intra, label="Intra RA", scan="S1"), rank,
            replace(FIXED, label="T0 fixed RAW", sparse=False), replace(rank, label="T0-RA RAW", sparse=False),
            replace(intra, label="Intra RA S2", scan="S2"), replace(rank, label="T0-RA S2", scan="S2", score_basis="temporal")]
    for setting in main:
        for cap in common if setting.label in ("Intra fixed", "T0 fixed", "Intra RA", "T0-RA",
                                               "Intra RA S2", "T0-RA S2") else practical:
            yield Arm(setting, cap)
    variants = [replace(rank, label=f"T0-RA {base}", base=base) for base in ("B1", "B3")]
    variants += [replace(rank, label="T0-RA INT8", bits=8),
                 replace(rank, label="T0-RA innovation", score_basis="temporal")]
    variants += [replace(rank, label=f"T0-RA block{size}", block_size=size) for size in (64, 128)]
    variants += [replace(rank, label=f"T0-RA refresh{n}", refresh=n) for n in (16, 64)]
    variants += [replace(rank, label="T0-RA scene", scene=True)]
    for setting in variants:
        for cap in low:
            yield Arm(setting, cap)


def _profile_cases(arms: list[Arm], assets: Assets):
    return {setting: assets.profile(setting) for setting in dict.fromkeys(arm.setting for arm in arms)}


def frame_evidence(out: Delivered, link: Link, target: np.ndarray, scale: np.ndarray) -> dict:
    pred = prediction_coordinates(out, link.profile)
    display = out.display
    return {
        "event": out.event, "frame_type": out.frame_type, "actual_bytes": out.actual_bytes,
        "received_bytes": out.received_bytes, "base_bytes": out.encoded.base_bytes,
        "sent_components": out.encoded.transmitted_symbols,
        "clipped_components": out.encoded.clipped_symbols,
        "visible_tokens": int(display.support.sum()) if display is not None else 0,
        "encoder_reference_digest": out.encoded.base.digest.hex(),
        "receiver_reference_digest": out.receiver.digest.hex() if out.receiver is not None else None,
        "base_support_digest": sha256(out.encoded.base.support.tobytes()).hexdigest(),
        "prediction_rms": rms(pred * scale), "residual_rms": rms((target.astype(np.int32) - pred) * scale),
        "reconstruction_rms": rms(display.coordinates * scale) if display is not None else None,
        "encode_ms": out.encoding_ms, "receive_ms": out.reception_ms,
    } | out.ledger


def _summary(rows: list[dict]) -> list[dict]:
    table = []
    fields = ("method", "base", "scan", "predictor", "sparse", "bits", "block_size", "refresh", "scene",
              "bandwidth", "cap_bytes", "payload_cap", "basis")
    for keys, items in groups(rows, fields):
        payload = sum(r["payload_bytes"] for r in items)
        total = sum(r["total_bytes"] for r in items)
        table.append(keys | summarize(items) | wire_rate(payload, total, len(items)) | {
            "maximum_frame_total_bytes": max(r["actual_bytes"] for r in items),
            "maximum_frame_payload_bytes": max(r["payload_bytes"] for r in items),
            "i_frames": sum(r["frame_type"] == "I" for r in items),
            "reference_mismatches": sum(r["encoder_reference_digest"] != r["receiver_reference_digest"]
                                        for r in items),
            "clipping_fraction": sum(r["clipped_components"] for r in items) / sum(r["sent_components"] for r in items),
            "tokens_per_frame": np.mean([r["visible_tokens"] for r in items]),
            "components_per_frame": np.mean([r["sent_components"] for r in items]),
        })
    return table


def _sample(clip, frame, cap, setting, rgb, pixels, output, manifest, binding, actual_bytes) -> None:
    chosen = {("DAVIS2017:blackswan", 25), ("DAVIS2017:car-roundabout", 32), ("DAVIS2017:dog", 48)}
    labels = {"Intra RA", "T0-RA", "T0-RA S2", "T0 fixed"}
    if (clip.video, frame) not in chosen or cap.bandwidth not in ("5", "10") or setting.label not in labels:
        return
    slug = clip.video.rsplit(":", 1)[-1]
    folder = output / "samples" / f"{slug}-{frame}"
    label = f"{setting.label.replace(' ', '_')}_{cap.bandwidth}Mbps"
    if not (folder / "source.png").is_file():
        save_native(folder, "source", rgb)
    save_native(folder, label, pixels)
    manifest.append({"video": clip.video, "frame": frame, "cap_bytes": cap.total, "actual_bytes": actual_bytes,
                     "method": setting.label, "image": str((folder / f"{label}.png").relative_to(output)),
                     "profile_binding_sha256": binding.hex()})


@torch.inference_mode()
def run(v3: Path, output: Path) -> None:
    if json_at(output / "diagnostic_reference_audit.json")["status"] != "PASS":
        raise RuntimeError("Phase 0 diagnostics and independent reference checks must pass")
    inputs = open_inputs(v3)
    assets = Assets(output)
    synth = assets.synthesis(inputs.model)
    scale = assets.arrays["scale"][LAYOUT.ranks]
    metric = NativeQuality()
    arms = list(experiments(assets.config))
    profile_cases = _profile_cases(arms, assets)
    implementation = {str(path.relative_to(ROOT)): sha256_file(path)
                      for path in [Path(__file__), *sorted((ROOT / "hla_tok/temporal").glob("*.py")),
                                   ROOT / "hla_tok/streaming/temporal.py", ROOT / "hla_tok/streaming/temporal_wire.py",
                                   Path(__file__).with_name("engine.py"), Path(__file__).with_name("inputs.py")]}
    rows: list[dict] = []
    samples: list[dict] = []
    allocations: dict[tuple[str, str], np.ndarray] = defaultdict(lambda: np.zeros(RANK_COMPONENT_SHAPE, dtype=np.int64))
    for clip in validation_plan(inputs)[0]:
        links = {arm: Link(profile_cases[arm.setting]) for arm in arms}
        last: dict[Arm, tuple[torch.Tensor, torch.Tensor]] = {}
        for i in range(clip.length):
            rgb = source_frame(inputs.reader, clip, i)
            target = target_lattice(project_source(inputs, assets.transform, rgb), scale)
            metric.prepare(rgb)
            scores: dict[bytes, tuple[torch.Tensor, dict]] = {}
            for arm, link in links.items():
                cap, setting = arm.cap, arm.setting
                result = link.send(target, i, cap.total, payload_cap=cap.payload)
                assert result.display is not None
                if result.encoded.raw not in scores:
                    with compute("bf16"):
                        image = synth.render(result.display)
                    scores[result.encoded.raw] = image, metric.score(image)
                image, quality = scores[result.encoded.raw]
                fields = asdict(setting)
                fields["method"] = fields.pop("label")
                row = fields | {"video": clip.video, "frame": i, "bandwidth": cap.bandwidth,
                                "cap_bytes": cap.total, "payload_cap": cap.payload, "basis": cap.basis,
                                "temporal_difference_mse": temporal_error(rgb, image, last.get(arm))}
                last[arm] = rgb, image
                evidence = frame_evidence(result, link, target, scale)
                rows.append(row | quality | evidence)
                sent = link.profile.scan[:result.encoded.transmitted_symbols]
                native_rank = LAYOUT.ranks[sent // LAYOUT.components]
                coordinates = native_rank * LAYOUT.components + sent % LAYOUT.components
                if setting.label in ("Intra RA", "T0-RA", "T0-RA S2") and cap.basis == "total_wire":
                    allocations[(setting.label, cap.bandwidth)] += np.bincount(
                        coordinates, minlength=int(np.prod(RANK_COMPONENT_SHAPE))).reshape(RANK_COMPONENT_SHAPE)
                _sample(clip, i, cap, setting, rgb, image, output, samples, link.profile.binding, result.actual_bytes)
            if (i + 1) % 10 == 0:
                print(f"Matrix {clip.video} {i + 1}/{clip.length}, {len(arms)} cases", flush=True)
        print(f"Matrix complete {clip.video}: {clip.length}", flush=True)
    summary = _summary(rows)
    write_csv(output / "validation_per_frame.csv", rows)
    write_csv(output / "ablation_comparison.csv", summary)
    write_csv(output / "rank_component_allocation.csv", [
        {"method": method, "bandwidth": band, "rank": r, "component": c,
         "mean_transmitted_regions": float(values[r, c] / sum(inputs.cohort.values())),
         "fraction_of_actual_regions": float(values[r, c] / sum(inputs.cohort.values()) / len(LAYOUT.tokens_for_rank(r)))}
        for (method, band), values in allocations.items() for r, c in np.ndindex(*RANK_COMPONENT_SHAPE)])
    write_json(output / "samples/manifest.json", {"checkpoint_sha256": inputs.checkpoint["sha256"],
              "lattice_sha256": assets.identity["lattice_sha256"], "cases": samples})
    write_json(output / "matrix_identity.json", {
        "config_sha256": sha256_file(CONFIG), "cases": [asdict(arm) for arm in arms],
        "profiles": [asdict(setting) | {"binding": profile.binding.hex(), "base_records": profile.base_records,
                     "base_components": profile.base_components, "max_base_bytes": profile.max_base_bytes}
                     for setting, profile in profile_cases.items()],
        "implementation": implementation,
        "test_pixel_reads": inputs.reader.counts["test"],
    })
    inputs.finish(output / "matrix_inputs.json")


if __name__ == "__main__":
    parser = run_arguments(__doc__)
    parser.set_defaults(output=RESULTS)
    options = parser.parse_args()
    run(options.v3, options.output)
