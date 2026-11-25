"""Serial single-H100 benchmarks through analysis, transmitted bytes and native synthesis."""

import argparse
import resource
from collections import defaultdict
from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
import torch

import hla_tok.streaming.temporal as streaming
from hla_tok.evaluation.io import environment, write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.streaming.continuous import encode_continuous
from hla_tok.temporal.layout import LAYOUT
from hla_tok.temporal.native_wire import NativeWire
from hla_tok.temporal.quantization import target_lattice
from hla_tok.temporal.state import BaseReferenceState
from scripts.hla_tok.continuous.benchmark import latency_summary, timed
from scripts.hla_tok.continuous.inputs import open_inputs
from scripts.hla_tok.continuous.matrix import compute
from scripts.hla_tok.temporal.inputs import RESULTS, Assets, Setting, source_frame, validation_plan
from scripts.hla_tok.v2.protocol import ROOT, json_at

GPU_SPANS = {"spatial_encoder": ("start", "spatial"), "aqt": ("spatial", "token"),
             "rank_transform": ("token", "projected"), "local_recovery_and_lift": ("coords", "features"),
             "d0_rgb": ("features", "rgb")}


class ProfileStages:
    """Time Python stages in a separate pass to keep per-record probes out of the serial latency."""

    def __init__(self):
        self.times = defaultdict(float)

    def _observe(self, operation, name, *, consume=False):
        def tracked(*args, **kwargs):
            start = perf_counter()
            result = operation(*args, **kwargs)
            if consume:
                result = list(result)
            self.times[name] += (perf_counter() - start) * 1000
            return result
        return tracked

    def install(self, stack: ExitStack, wire: NativeWire | None) -> None:
        methods: tuple[tuple[object, str, str], ...] = (
            (streaming, "_encoding_header", "reference_prediction"),
            (streaming, "predict", "reference_prediction"),
            (streaming, "quantize_symbols", "residual_quantization"),
            (streaming.Blocks, "__init__", "sparse_packing"),
            (streaming.Blocks, "payloads", "sparse_packing"),
            (streaming, "record_bytes", "serialize"),
            (streaming, "open_header", "deserialize"),
            (streaming, "open_record", "deserialize"),
            (streaming._Frame, "_decode_batch", "incremental_residual_decode"),
            (streaming, "commit", "base_state_commit"),
        )
        if wire is not None:
            methods += ((NativeWire, "pack", "native_sparse_pack_and_serialize"),
                        (NativeWire, "open", "native_deserialize_and_residual_decode"))
        for owner, name, phase in methods:
            stack.enter_context(patch.object(owner, name, self._observe(
                getattr(owner, name), phase, consume=name == "payloads")))


class PathUnderTest:
    def __init__(self, inputs, synth, profile, scale, cap, wire: NativeWire | None = None):
        self.model = inputs.model
        self.synth = synth
        self.profile = profile
        self.cap = cap
        self.scale = scale
        self.reference: BaseReferenceState | None = None
        self.wire = wire
        self.receiver = streaming.TemporalReceiver(profile, wire=wire)
        keys = {mark for pair in GPU_SPANS.values() for mark in pair}
        self.events = {name: torch.cuda.Event(enable_timing=True) for name in keys}
        self.hook = self.model.encoder.register_forward_hook(self._spatial_end)

    def _spatial_end(self, _module, _inputs, _result) -> None:
        self.events["spatial"].record()

    def reset(self) -> None:
        self.reference = None
        self.receiver.reset_reference()

    def frame(self, source, frame_id) -> dict:
        ev = self.events
        start = perf_counter()
        ev["start"].record()
        tokens = encode_continuous(self.model, source)
        ev["token"].record()
        projected = self.synth.transform.project(tokens)[0]
        ev["projected"].record()
        ev["projected"].synchronize()
        host, gpu_to_host = timed(lambda: projected.cpu().numpy())
        target, lattice_ms = timed(lambda: target_lattice(host, self.scale))
        sent, send_ms = timed(lambda: streaming.encode_frame(
            target, frame_id, self.reference, self.profile, self.cap, wire=self.wire))
        self.reference = sent.base
        display, receive_ms = timed(lambda: self._receive(sent, frame_id))
        coordinates, host_to_gpu = timed(lambda: self.synth.receive_coordinates(display))
        ev["coords"].record()
        features = self.synth.recover(coordinates, display.support)
        ev["features"].record()
        self.synth.reconstruct(features)
        ev["rgb"].record()
        ev["rgb"].synchronize()
        latency = (perf_counter() - start) * 1000
        gpu = {stage: ev[a].elapsed_time(ev[b]) for stage, (a, b) in GPU_SPANS.items()}
        return gpu | {"gpu_to_host": gpu_to_host, "host_to_gpu": host_to_gpu, "lattice_rounding": lattice_ms,
                      "encode_application_wire": send_ms, "receive_application_wire": receive_ms,
                      "total_serial": latency, "application_bytes": len(sent.raw)}

    def _receive(self, sent, frame_id):
        if len(sent.raw) > self.cap:
            raise RuntimeError("timed frame exceeded its serialized byte budget")
        self.receiver.receive_stream(sent.raw)
        observed = self.receiver.commit_base(frame_id)
        if observed.digest != sent.base.digest:
            raise RuntimeError("timed frame reference digests differ")
        display = self.receiver.render_prefix(frame_id)
        self.receiver.finalize_frame(frame_id)
        return display


def _cycle(path: PathUnderTest, banks: list[list[torch.Tensor]], count: int):
    position = 0
    while position < count:
        for frames in banks:
            path.reset()
            for frame_id, source in enumerate(frames[:count - position]):
                yield path.frame(source, frame_id)
                position += 1


def _bench(path: PathUnderTest, banks: list, knobs: dict, precision: str, scope: dict) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    collected, repeat_fps = [], []
    with compute(precision):
        for repeat in range(knobs["repeats"]):
            for _ in _cycle(path, banks, knobs["warmup"]):
                pass
            begin = perf_counter()
            rows = list(_cycle(path, banks, knobs["frames"]))
            repeat_fps.append(knobs["frames"] / (perf_counter() - begin))
            collected.extend(rows)
            print(f"{precision} {scope['bandwidth']} Mbps repeat {repeat + 1}: {repeat_fps[-1]:.2f} FPS", flush=True)
    summary = latency_summary([row["total_serial"] for row in collected])
    fps = len(repeat_fps) / sum(1 / value for value in repeat_fps)
    memory = {"peak_allocated_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
              "peak_reserved_gpu_mib": torch.cuda.max_memory_reserved() / 2**20,
              "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
              "base_reference_bytes": len(path.reference._coordinates) + len(path.reference._support)
              + len(path.reference._previous) if path.reference is not None else 0,
              "maximum_enhancement_cache_bytes": len(path.profile.scan) - path.profile.base_components,
              "max_display_coordinates_bytes": LAYOUT.token_count * (LAYOUT.components * 2 + 1)}
    rates = scope | summary | memory | {
        "measured_frames": len(collected), "warmup_per_repeat": knobs["warmup"], "repeat_fps": repeat_fps,
        "steady_state_fps": fps, "mean_application_bytes": float(np.mean([row["application_bytes"] for row in collected])),
        "pass_30fps": fps >= knobs["min_fps"], "pass_50fps": fps >= knobs["target_fps"],
        "pass_p95": summary["p95_ms"] <= knobs["max_p95_ms"],
    }
    times = [scope | {"stage": label, "timing": "cuda_event" if label in GPU_SPANS else "wall"}
             | latency_summary([row[label] for row in collected])
             for label in collected[0] if label != "application_bytes"]
    return {"runtime": rates, "stages": times, "per_frame": collected}


def _profiling_pass(path: PathUnderTest, banks: list, precision: str, scope: dict) -> list[dict]:
    profiler = ProfileStages()
    measurements = []
    with ExitStack() as stack, compute(precision):
        profiler.install(stack, path.wire)
        for _ in _cycle(path, banks, 16):
            profiler.times.clear()
        for _ in _cycle(path, banks, 64):
            measurements.append(dict(profiler.times))
            profiler.times.clear()
    return [scope | {"stage": stage, "timing": "in-process wall stage probe, separate 64-frame pass"}
            | latency_summary([row.get(stage, 0) for row in measurements])
            for stage in sorted({key for row in measurements for key in row})]


def _endpoint_roundtrip(target: np.ndarray, frame_id: int, receiver, reference, cap: int, profile, wire):
    sent = streaming.encode_frame(target, frame_id, reference, profile, cap, wire=wire)
    receiver.receive_stream(sent.raw)
    display = receiver.render_prefix(frame_id)
    ledger = receiver.wire_ledger(frame_id)
    state = receiver.commit_base(frame_id)
    receiver.finalize_frame(frame_id)
    return sent, state, display, ledger


def _compare_backends(inputs, synth, profile, scale, frames, wire: NativeWire, caps: dict,
                      precisions: list[str], binding: dict, output: Path) -> None:
    parity = {}
    for precision in precisions:
        endpoints = {b: (streaming.TemporalReceiver(profile), streaming.TemporalReceiver(profile, wire=wire))
                     for b in caps}
        fingerprints = {b: sha256() for b in caps}
        histories: dict[str, tuple[BaseReferenceState | None, BaseReferenceState | None]] = {
            b: (None, None) for b in caps
        }
        with compute(precision):
            for i, source in enumerate(frames):
                values = synth.transform.project(encode_continuous(inputs.model, source))[0].cpu().numpy()
                q = target_lattice(values, scale)
                for band, (standard, fast) in endpoints.items():
                    a, b = (_endpoint_roundtrip(q, i, receiver, prior, caps[band], profile, endpoint)
                            for (receiver, endpoint), prior in zip(
                                ((standard, None), (fast, wire)), histories[band], strict=True))
                    histories[band] = a[0].base, b[0].base
                    raw_matches = a[0].raw == b[0].raw and a[0].clipped_symbols == b[0].clipped_symbols
                    states_match = a[0].base.digest == b[0].base.digest == a[1].digest == b[1].digest
                    display_matches = np.array_equal(a[2].coordinates, b[2].coordinates) and np.array_equal(
                        a[2].support, b[2].support)
                    if not (raw_matches and states_match and display_matches and a[3] == b[3]):
                        raise RuntimeError(f"Record backends disagree on {precision} {band} Mbps frame {i}")
                    if i in (0, len(frames) - 1) and not torch.equal(synth.render(a[2]), synth.render(b[2])):
                        raise RuntimeError("Record backends produced different native RGB")
                    fingerprints[band].update(a[0].raw)
        parity[precision] = {budget: digest.hexdigest() for budget, digest in fingerprints.items()}
    write_json(output / "native_wire_parity.json", {
        "status": "PASS", "source_frames": len(frames), "checks": len(frames) * len(caps) * len(precisions),
        "scope": "independent sender/receiver histories with byte, ledger, state and integer display equality",
        "native_rgb_first_and_last_frame_equality": True, "stream_sha256": parity, "binding": binding,
    })


@torch.inference_mode()
def run(v3: Path, output: Path, *, short: bool = False, fixed_setting: str | None = None,
        wire_library: Path | None = None) -> None:
    inputs = open_inputs(v3)
    assets = Assets(output)
    selected = json_at(output / "candidate_selection.json")
    setting = Setting(**selected["setting"])
    if fixed_setting is not None:
        selected_settings = json_at(output / "matrix_identity.json")["profiles"]
        row, = (item for item in selected_settings if item["label"] == fixed_setting)
        setting = Setting(**{key: row[key] for key in Setting.__dataclass_fields__})
    profile = assets.profile(setting)
    wire = NativeWire(wire_library, profile) if wire_library is not None else None
    synth = assets.synthesis(inputs.model)
    scale = assets.arrays["scale"][LAYOUT.ranks]
    clips = [clip for clip in validation_plan(inputs)[0] if clip.length >= 64]
    banks = [[source_frame(inputs.reader, clip, frame) for frame in range(clip.length)] for clip in clips]
    knobs = assets.config["runtime"]
    suffix = "_native" if wire is not None else ""
    folder = output / (("runtime_smoke" if short else "runtime_cases") + suffix)
    if short:
        knobs = knobs | {"warmup": 16, "frames": 64, "repeats": 1, "precisions": ["bf16"]}
    binding = {"hardware": environment(), "model_state_sha256": inputs.checkpoint["state_dict_sha256"],
               "profile_binding": profile.binding.hex(), "runtime_protocol": knobs,
               "clips": [{"video": clip.video, "frames": clip.length} for clip in clips],
               "sources_sha256": {str(path.relative_to(ROOT)): sha256_file(path)
                                  for path in [Path(__file__), *sorted((ROOT / "hla_tok/temporal").glob("*.py")),
                                               ROOT / "hla_tok/streaming/temporal.py"]}}
    binding["wire_backend"] = ("native C11" if wire is not None else "python numpy")
    if wire is not None:
        binding["native_library_sha256"] = sha256_file(wire.path)
        binding["native_source_sha256"] = sha256_file(ROOT / "hla_tok/temporal/records.c")
    if wire is not None and not short:
        bands = {band: assets.config["wire_caps"][band] for band in ("5", "10", "20", "40")}
        _compare_backends(inputs, synth, profile, scale, banks[0], wire, bands, knobs["precisions"], binding, output)
    runtimes, stages = [], []
    for precision in knobs["precisions"]:
        for bandwidth in ("5", "10", "20", "40"):
            cap = assets.config["wire_caps"][bandwidth]
            scope = {"method": setting.label, "precision": precision, "bandwidth": bandwidth, "cap_bytes": cap}
            cache = folder / f"{precision}_{bandwidth}.json"
            if cache.exists():
                archived = json_at(cache)
                if archived["identity"] != binding:
                    raise RuntimeError(f"stale runtime checkpoint {cache}")
                result = archived["result"]
            else:
                path = PathUnderTest(inputs, synth, profile, scale, cap, wire)
                try:
                    result = _bench(path, banks, knobs, precision, scope)
                    result["stages"].extend(_profiling_pass(path, banks, precision, scope))
                finally:
                    path.hook.remove()
                write_json(cache, {"identity": binding, "result": result})
            runtimes.append(result["runtime"])
            stages.extend(result["stages"])
    write_csv(folder / "latency_breakdown.csv", stages)
    if not short:
        write_csv(output / f"latency_breakdown{suffix}.csv", stages)
        production = [row for row in runtimes if row["precision"] == "bf16"]
        write_json(output / f"runtime_audit{suffix}.json", binding | {
            "status": "PASS" if all(row["pass_30fps"] and row["pass_p95"] for row in production) else "FAIL",
            "formal_conditions": runtimes, "scope": "GPU native input to GPU RGB, files and network excluded",
            "notes": "Same causal native clip repeated with independent state reset at every loop boundary.",
            "serialized_application_wire": True, "motion_estimation": "NOT_APPLICABLE",
            "deployment_calibration_asset_bytes": assets.path.stat().st_size,
            "profile_scan_bytes_loaded": profile.scan.nbytes,
            "model_compiled": False,
            "compiled_wire_record_backend": wire is not None,
        })
        inputs.finish(output / f"runtime_inputs{suffix}.json")


if __name__ == "__main__":
    from scripts.hla_tok.continuous.inputs import run_arguments

    parser: argparse.ArgumentParser = run_arguments(__doc__)
    parser.set_defaults(output=RESULTS)
    parser.add_argument("--short", action="store_true")
    parser.add_argument("--fixed-setting")
    parser.add_argument("--wire-library", type=Path)
    args = parser.parse_args()
    run(args.v3, args.output, short=args.short, fixed_setting=args.fixed_setting, wire_library=args.wire_library)
