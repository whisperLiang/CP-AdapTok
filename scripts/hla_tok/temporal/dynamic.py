"""Long, native closed loops over changing budgets and physical record loss."""

import subprocess
import sys
from dataclasses import asdict, replace
from hashlib import sha256
from pathlib import Path
from random import Random
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import Tensor

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.evaluation.reconstruction import save_native
from hla_tok.streaming.continuous import ContinuousReceiver, RankOrder, encode_continuous
from hla_tok.streaming.temporal import TemporalReceiver, encode_frames
from hla_tok.streaming.temporal_wire import HEADER_BYTES, open_record
from hla_tok.temporal.layout import LAYOUT
from hla_tok.temporal.quantization import target_lattice
from scripts.hla_tok.continuous.inputs import open_inputs, run_arguments
from scripts.hla_tok.continuous.matrix import compute, groups
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.temporal.engine import Link, temporal_error
from scripts.hla_tok.temporal.evaluate import frame_evidence
from scripts.hla_tok.temporal.inputs import (
    RESULTS,
    Assets,
    Setting,
    project_source,
    source_frame,
    validation_plan,
)
from scripts.hla_tok.temporal.receive import rgb_digest
from scripts.hla_tok.v2.protocol import ROOT, json_at


def _channels(length: int, seed: int) -> dict[str, list[tuple[int, str]]]:
    rng = Random(seed)
    levels = (5, 10, 20, 40)
    fixed = {f"fixed_{rate}": [(rate, "full")] * length for rate in levels}
    segment = [min(2, 3 * i // length) for i in range(length)]
    switches = {"5-40-5": [(5, 40, 5)[a] for a in segment], "40-5-40": [(40, 5, 40)[a] for a in segment],
                "random": [rng.choice(levels) for _ in range(length)]}
    return fixed | {name: [(rate, "full") for rate in values] for name, values in switches.items()} | {
        "base_only": [(5, "base-only")] * length,
        "periodic_enhancement_loss": [(40, "lose-enhancement" if i % 5 == 3 else "full") for i in range(length)],
        "all_enhancement_lost": [(40, "lose-enhancement")] * length,
        "base_loss_wait_I": [(10, "lose-base" if i == 7 else "full") for i in range(length)],
    }


def _save_received_samples(output: Path, clipboard: list, setting: Setting, clip, index,
                           rgb, pixels, out, condition, cap, digest) -> None:
    if clip.video != "DAVIS2017:soapbox" or index != clip.length - 1 or condition not in ("fixed_5", "fixed_40"):
        return
    sample = output / "samples/soapbox-98"
    save_native(sample, "source", rgb)
    label = f"{setting.label.replace(' ', '_')}_{condition}"
    save_native(sample, label, pixels)
    clipboard.append({"video": clip.video, "frame": index, "cap_bytes": cap, "actual_bytes": out.actual_bytes,
                      "method": setting.label, "image": str((sample / f"{label}.png").relative_to(output)),
                      "profile_binding_sha256": digest.hex(), "event": out.event})


def _within_prefix(receiver, result, small) -> None:
    index = receiver.receive_stream(result.encoded.raw)
    stop, record_count = HEADER_BYTES, 0
    while stop < len(small.encoded.raw):
        _, stop = open_record(result.encoded.raw, stop)
        record_count += 1
    if not result.encoded.raw.startswith(small.encoded.raw) or stop != len(small.encoded.raw):
        raise RuntimeError("independent physical wire prefixes differ")
    from_high = receiver.render_prefix(index, record_count)
    if small.display is None or not np.array_equal(from_high.coordinates, small.display.coordinates):
        raise RuntimeError("independently received integer prefixes differ")
    if not np.array_equal(from_high.support, small.display.support):
        raise RuntimeError("independently received support prefixes differ")
    receiver.finalize_frame(index)


@torch.inference_mode()
def run(v3: Path, output: Path) -> None:
    inputs = open_inputs(v3)
    assets = Assets(output)
    selected = json_at(output / "candidate_selection.json")
    setting = Setting(**selected["setting"])
    profile = assets.profile(setting)
    synth = assets.synthesis(inputs.model)
    caps = {int(k): assets.config["wire_caps"][k] for k in ("5", "10", "20", "40")}
    metric = NativeQuality()
    scale = assets.arrays["scale"][LAYOUT.ranks]
    primary, additional = validation_plan(inputs)
    clips = [clip for clip in primary if clip.length >= 64] + additional
    rows: list[dict] = []
    wire_prefix_checks: list[dict] = []
    sample_rows: list[dict] = []
    subprocess_cases: list[tuple] = []
    boundary_frames = []
    for clip in clips:
        channels = _channels(clip.length, assets.config["seed"])
        links = {name: Link(profile) for name in channels}
        check_prefix = TemporalReceiver(profile)
        last: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        for i in range(clip.length):
            rgb = source_frame(inputs.reader, clip, i)
            analysis = project_source(inputs, assets.transform, rgb)
            target = target_lattice(analysis, scale)
            metric.prepare(rgb)
            scored = {}
            current = {}
            for name, plan in channels.items():
                rate, path = plan[i]
                out = links[name].send(target, i, caps[rate], channel=path)
                current[name] = out
                row = {"video": clip.video, "frame": i, "condition": name, "bandwidth": rate,
                       "cap_bytes": caps[rate], "channel_event": path}
                evidence = frame_evidence(out, links[name], target, scale)
                if out.display is None:
                    last.pop(name, None)
                    rows.append(row | evidence | {"quality_status": "WAIT_I"})
                    continue
                key = out.encoded.raw[:out.received_bytes]
                if key not in scored:
                    with compute("bf16"):
                        pixels = synth.render(out.display)
                    scored[key] = pixels, metric.score(pixels)
                pixels, scores = scored[key]
                delta = temporal_error(rgb, pixels, last.get(name))
                last[name] = rgb, pixels
                rows.append(row | evidence | scores | {"quality_status": "MEASURED", "temporal_difference_mse": delta})
                _save_received_samples(output, sample_rows, setting, clip, i, rgb, pixels, out, name,
                                       caps[rate], profile.binding)
                if clip == clips[0] and i in (0, 1, 2) and name == "fixed_10":
                    subprocess_cases.append((out, rgb_digest(pixels)))
            for name, out in current.items():
                if name == "base_loss_wait_I":
                    continue
                if out.encoded.base.digest != current["fixed_40"].encoded.base.digest or out.receiver is None:
                    raise RuntimeError("dynamic enhancement affected committed history")
            for lo, hi in (("fixed_5", "fixed_10"), ("fixed_10", "fixed_20"), ("fixed_20", "fixed_40")):
                if not current[hi].encoded.raw.startswith(current[lo].encoded.raw):
                    raise RuntimeError("dynamic wire prefixes are not strict")
            _within_prefix(check_prefix, current["fixed_40"], current["fixed_5"])
            wire_prefix_checks.append({"video": clip.video, "frame": i, "base_sha256": current["fixed_5"].encoded.base.digest.hex(),
                                       "low_bytes": current["fixed_5"].actual_bytes,
                                       "high_bytes": current["fixed_40"].actual_bytes})
            if i in (0, clip.length - 1):
                boundary_frames.append((clip.video, i, target, rgb))
            if (i + 1) % 16 == 0:
                print(f"Dynamic {clip.video} {i + 1}/{clip.length}", flush=True)
    _artifacts(output, rows, clips, wire_prefix_checks, subprocess_cases, boundary_frames, setting, inputs, assets)
    sample_rows.extend(_scene_samples(output, assets, setting, boundary_frames, synth))
    samples = json_at(output / "samples/manifest.json")
    samples["cases"].extend(r for r in sample_rows if r not in samples["cases"])
    for case in samples["cases"]:
        image_file = output / case["image"]
        case["image_sha256"] = sha256_file(image_file)
        case["source_png_sha256"] = sha256_file(image_file.with_name("source.png"))
    write_json(output / "samples/manifest.json", samples)
    inputs.finish(output / "dynamic_inputs.json")


def _causality(inputs, assets, output, setting, frames) -> None:
    _, _, first_target, rgb = frames[0]
    profile = assets.profile(setting)
    accessed = []

    def guarded_frame():
        accessed.append(0)
        full = project_source(inputs, assets.transform, rgb)
        yield target_lattice(full, assets.arrays["scale"][LAYOUT.ranks]), 20833
        raise RuntimeError("future frame was read by the encoder")

    present = next(encode_frames(guarded_frame(), profile))
    _, _, _, future_rgb = frames[1]
    independent = []
    for raw_next in (future_rgb, 1 - future_rgb):
        def frames_with_future():
            yield first_target, 20833
            projection = project_source(inputs, assets.transform, raw_next)
            yield target_lattice(projection, assets.arrays["scale"][LAYOUT.ranks]), 166666
        independent.append(next(encode_frames(frames_with_future(), profile)))
    if accessed != [0] or any(present.raw != frame.raw for frame in independent):
        raise RuntimeError("native current wire changed with future content")
    write_json(output / "causality_audit.json", {
        "status": "PASS", "native_shape_hw": [1080, 1920], "algorithmic_lookahead_frames": 0,
        "inputs_before_yield": len(accessed), "future_frame_changes": 1,
        "present_wire_sha256": sha256(present.raw).hexdigest(), "present_base_sha256": present.base.digest.hex(),
        "boundaries": [{"video": v, "frame": f} for v, f, _, _ in frames],
    })


def _separate_process(inputs, output, setting, cases) -> dict:
    with TemporaryDirectory(prefix="hlt-receive-") as work:
        root = Path(work)
        write_json(root / "setting.json", asdict(setting))
        packets = []
        for index, (result, _) in enumerate(cases):
            destination = root / f"{index}.hlt"
            destination.write_bytes(result.encoded.raw)
            packets.append(destination)
        subprocess.run([sys.executable, "-m", "scripts.hla_tok.temporal.receive", "--assets", str(output),
                        "--checkpoint", inputs.checkpoint["verified_path"], "--setting", str(root / "setting.json"),
                        "--output", str(root / "audit.json"), *map(str, packets)], cwd=ROOT, check=True)
        observed = json_at(root / "audit.json")
    expected = [(row.encoded.base.digest.hex(), digest) for row, digest in cases]
    actual = [(row["base_sha256"], row["rgb_float32_sha256"]) for row in observed["frames"]]
    if observed["has_analysis_modules"] or expected != actual:
        raise RuntimeError("separate receiving process disagreed on base or native RGB")
    return observed


def _scene_samples(output, assets, setting, frames, synth) -> list[dict]:
    scene_profile = assets.profile(replace(setting, scene=True))
    folder = output / "samples/scene-boundary"
    left, right = frames[1], frames[2]
    link = Link(scene_profile)
    link.send(left[2], 0, 20833)
    frame = link.send(right[2], 1, 20833)
    if frame.display is None:
        raise RuntimeError("scene boundary could not be synthesized")
    with compute("bf16"):
        image = synth.render(frame.display)
    for name, value in (("previous_source", left[3]), ("source", right[3]), ("reconstruction", image)):
        save_native(folder, name, value)
    return [{"video": right[0], "frame": right[1], "previous_video": left[0], "previous_frame": left[1],
             "cap_bytes": 20833, "actual_bytes": frame.actual_bytes, "method": setting.label,
             "image": "samples/scene-boundary/reconstruction.png", "event": "source switch engineering audit",
             "frame_type": frame.frame_type, "profile_binding_sha256": scene_profile.binding.hex()}]


def _bypass(inputs, assets, frames) -> list[dict]:
    synth = assets.synthesis(inputs.model)
    frozen = ContinuousReceiver.from_model(inputs.model)
    mask = np.ones(LAYOUT.token_count, dtype=bool)
    baseline = RankOrder.native("cuda")
    rows = []
    for precision in ("bf16", "fp32"):
        for video, index, _, rgb in frames[::2]:
            with compute(precision):
                coefficients: Tensor = encode_continuous(inputs.model, rgb)
                reference = frozen.decode(coefficients, baseline).float()
                reconstruction = synth.render_coefficients(assets.transform.project(coefficients), mask).float()
            expected = (rgb - reference).square().mean()
            observed = (rgb - reconstruction).square().mean()
            shift = float(10 * torch.log10(expected / observed))
            if abs(shift) > 0.05:
                raise RuntimeError("full-support PCA bypass exceeded the frozen C0 PSNR tolerance")
            rows.append({"video": video, "frame": index, "precision": precision, "psnr_delta_db": shift,
                         "maximum_rgb_difference": float((reconstruction - reference).abs().max())})
    return rows


def _stress_checks(assets, setting, frames) -> dict:
    static_profile = assets.profile(replace(setting, scene=False))
    still = Link(static_profile)
    continuous = [still.send(frames[0][2], i, 20833) for i in range(70)]
    exact = all(out.receiver is not None and out.receiver.digest == out.encoded.base.digest
                for out in continuous)
    steady = all(np.array_equal(a.encoded.base.coordinates, b.encoded.base.coordinates)
                 for a, b in zip(continuous, continuous[1:])
                 if b.encoded.frame_id - b.encoded.base.last_intra >= 8)
    if not exact or not steady:
        raise RuntimeError("stationary integer reference did not settle within eight P frames")
    scene_profile = assets.profile(replace(setting, scene=True))
    cases = []
    for left, right in zip(frames[1::2], frames[2::2]):
        link = Link(scene_profile)
        link.send(left[2], 0, 20833)
        new = link.send(right[2], 1, 20833)
        cases.append({"previous_source": left[0], "previous_index": left[1], "new_source": right[0],
                      "new_index": right[1], "type": new.frame_type, "bytes": new.actual_bytes,
                      "base_sha256": new.encoded.base.digest.hex()})
    return {"repeat_native_still_frames_for_engineering": 70, "stationary_integer_reference": steady,
            "source_switch_stress": cases, "source_switch_usage": "engineering stimulus, excluded from continuous metrics"}


def _artifacts(output, rows, clips, pairs, subprocess_cases, boundary_frames, setting, inputs, assets) -> None:
    write_csv(output / "closed_loop_per_frame.csv", rows)
    summaries: list[dict] = []
    events: list[dict] = []
    for keys, values in groups(rows, ("video", "condition")):
        valid = [row for row in values if row["quality_status"] == "MEASURED"]
        thirds = np.array_split(np.arange(len(values)), 3)
        lengths = [sum(values[i]["quality_status"] == "MEASURED" for i in span) for span in thirds]
        quality_by_third = [summarize([values[i] for i in span if values[i]["quality_status"] == "MEASURED"])
                            if n else {} for n, span in zip(lengths, thirds, strict=True)]
        summaries.append(keys | summarize(valid) | {"measured": len(valid), "total_frames": len(values),
                         "base_loss_frames": len(values) - len(valid), "mean_sent_bytes": np.mean(
                             [row["actual_bytes"] for row in values]), "wire_mbps_30fps": 240e-6 * np.mean(
                             [row["actual_bytes"] for row in values]),
                         "mean_received_bytes": np.mean([row["received_bytes"] for row in values]),
                         "p95_encode_receive_ms": np.percentile([row["encode_ms"] + row["receive_ms"]
                                                               for row in values], 95),
                         **{f"{seg}_psnr": score.get("global_psnr") for seg, score in zip(
                             ("early", "middle", "late"), quality_by_third, strict=True)}})
        if keys["condition"] == "base_loss_wait_I":
            events.extend(keys | {"frame": row["frame"], "event": row["event"]} for row in values
                          if row["event"] in {"base_record_lost", "WAIT_I", "I_recovery"})
    write_csv(output / "dynamic_bandwidth_results.csv", summaries)
    _causality(inputs, assets, output, setting, boundary_frames)
    wire = _separate_process(inputs, output, setting, subprocess_cases)
    write_json(output / "wire_audit.json", {"status": "PASS", "independent_process": wire,
              "per_frame_budgets": True, "scope": "codec application bytes",
              "network_layer_bytes": "NOT_RUN", "motion_records": 0, "full_support_c0_bypass": _bypass(inputs, assets, boundary_frames)})
    write_json(output / "prefix_audit.json", {"status": "PASS", "independently_encoded_video_frames": len(pairs),
              "caps": [20833, 41666, 83333, 166666], "integer_wire_and_support_checks": pairs})
    write_json(output / "reference_state_audit.json", {"status": "PASS", "frames": len(rows),
              "delivered_base_mismatches": 0, "base_recovery_events": events,
              "long_validation": {clip.video: clip.length for clip in clips},
              "requested_lengths": {"64": "PASS", "128": "NOT_AVAILABLE", "256": "NOT_AVAILABLE"},
              "engineering_stress_checks": _stress_checks(assets, setting, boundary_frames),
              "per_frame_sha256": sha256_file(output / "closed_loop_per_frame.csv")})


if __name__ == "__main__":
    parser = run_arguments(__doc__)
    parser.set_defaults(output=RESULTS)
    args = parser.parse_args()
    run(args.v3, args.output)
