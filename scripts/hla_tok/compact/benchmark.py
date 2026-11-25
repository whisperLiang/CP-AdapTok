"""Measure the real Compact RGB-to-RGB physical packet path on one H100."""

import resource
from collections.abc import Iterable
from itertools import product
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor

from hla_tok.compression.profiles import PROFILES_BY_LABEL
from hla_tok.evaluation.io import environment, write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.streaming.compact import CompactCodec
from hla_tok.streaming.compact_wire import CompactFrame, max_budget
from hla_tok.streaming.continuous import MAX_K, ContinuousReceiver, RankOrder, encode_continuous
from scripts.hla_tok.compact.assets import load_asset
from scripts.hla_tok.compact.inputs import CONFIG, OUTPUT, compact_inputs, compact_orders
from scripts.hla_tok.continuous.benchmark import latency_summary, timed
from scripts.hla_tok.continuous.inputs import Inputs, locked_output, run_arguments
from scripts.hla_tok.continuous.matrix import compute, representative, validation_frames
from scripts.hla_tok.v2.protocol import ROOT, json_at

SPANS = {
    "encoder": ("start", "spatial"),
    "tokenizer": ("spatial", "token"),
    "ordering": ("token", "ordered"),
    "projection": ("ordered", "projected"),
    "quantization": ("projected", "quantized"),
    "dequantization": ("received", "dequantized"),
    "lift": ("dequantized", "lifted"),
    "recovery": ("lifted", "grid"),
    "rgb_decoder": ("grid", "rgb"),
}
WALL_STAGES = ("gpu_to_cpu", "pack", "unpack", "cpu_to_gpu", "total_serial")


class TimedPath:
    def __init__(self, inputs: Inputs, codec: CompactCodec, budget: int):
        self.model = inputs.model
        self.receiver = ContinuousReceiver.from_model(inputs.model)
        self.codec = codec
        self.budget = budget
        self.events = {key: torch.cuda.Event(enable_timing=True) for names in SPANS.values() for key in names}
        self.packet_bytes = 0
        self.hook = inputs.model.encoder.register_forward_hook(self._spatial_event)

    def _spatial_event(self, _module, _inputs, _outputs):
        self.events["spatial"].record()

    def close(self) -> None:
        self.hook.remove()

    def _sender(self, source: Tensor) -> Tensor:
        e = self.events
        full = encode_continuous(self.model, source)
        e["token"].record()
        ordered = self.codec.order.apply(full)
        e["ordered"].record()
        u = self.codec.transform.encode(ordered, self.codec.ranks)
        e["projected"].record()
        codes = self.codec.encode_codes(u[:, :self.budget]).contiguous()
        e["quantized"].record()
        e["quantized"].synchronize()
        return codes

    def _exchange(self, codes: Tensor, frame_id: int) -> tuple[Tensor, dict]:
        cpu, d2h = timed(codes.cpu)
        raw, packed = timed(lambda: CompactFrame(self.codec.identity, frame_id, self.codec.pack(cpu))
                            .serialize_range(0, self.budget))
        decoded, unpacked = timed(lambda: self.codec.prefix_codes(raw))
        self.packet_bytes = len(raw)
        result, h2d = timed(lambda: decoded.to(codes.device))
        return result, dict(zip(WALL_STAGES, (d2h, packed, unpacked, h2d)))

    def _synthesis(self, codes: Tensor) -> None:
        e = self.events
        e["received"].record()
        u = self.codec.dequantize(codes)
        e["dequantized"].record()
        lifted = self.codec.lift(u)
        e["lifted"].record()
        grid = self.receiver.recover(lifted, self.codec.order)
        e["grid"].record()
        self.receiver.reconstruct(grid)
        e["rgb"].record()
        e["rgb"].synchronize()

    def frame(self, source: Tensor, frame_id: int) -> dict:
        start = perf_counter()
        self.events["start"].record()
        codes, wall = self._exchange(self._sender(source), frame_id)
        self._synthesis(codes)
        total = (perf_counter() - start) * 1000
        gpu = {key: self.events[a].elapsed_time(self.events[b]) for key, (a, b) in SPANS.items()}
        return wall | gpu | {"total_serial": total}


def _representative(path: TimedPath, frames: list[Tensor], config: dict) -> tuple[list[dict], list[float]]:
    rows: list[dict] = []
    fps: list[float] = []
    for _ in range(config["repeats"]):
        for i in range(config["warmup_frames"]):
            path.frame(frames[i % len(frames)], i)
        begin = perf_counter()
        count = config["measured_frames"]
        rows.extend(path.frame(frames[i % len(frames)], i) for i in range(count))
        fps.append(count / (perf_counter() - begin))
    return rows, fps


def _validation(path: TimedPath, sources: Iterable[tuple[str, int, Tensor]]) -> list[dict]:
    return [{"video": video, "frame": i} | path.frame(source, i) for video, i, source in sources]


def bench_assets(selection: dict, assets: dict) -> list[str]:
    ranked = [row["asset"] for row in selection["ranking"]]
    by_bits: dict[int, str] = {}
    for key in ranked:
        profile = PROFILES_BY_LABEL[assets[key]["profile"]]
        by_bits.setdefault(profile.bits, key)
    return list(dict.fromkeys([selection["main_asset"], *(by_bits[b] for b in (16, 4, 6, 8) if b in by_bits)]))


def budgets_for(config: dict, asset: dict) -> list[int]:
    profile = PROFILES_BY_LABEL[asset["profile"]]
    return sorted({MAX_K, *(max_budget(cap, profile) for cap in config["frame_wire_byte_caps"])})


def _measure(inputs: Inputs, codec: CompactCodec, condition: dict, frames: list, config: dict) -> dict:
    path = TimedPath(inputs, codec, condition["budget"])
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    try:
        with compute(condition["compute_dtype"]):
            rows, fps = _representative(path, frames, config)
    finally:
        path.close()
    summary = latency_summary([r["total_serial"] for r in rows])
    steady = len(fps) / sum(1 / value for value in fps)
    timing = condition | summary | {"steady_state_fps": steady, "packet_bytes": path.packet_bytes,
                                   "repeated_measurements": len(rows), "independent_source_frames": len(frames),
                                   "warmup_per_repeat": config["warmup_frames"],
                                   "repeat_1_fps": fps[0], "repeat_2_fps": fps[1], "repeat_3_fps": fps[2],
                                   "pass_30fps": steady >= config["minimum_fps"],
                                   "pass_60fps": steady >= config["ideal_fps"],
                                   "pass_p95": summary["p95_ms"] <= config["max_p95_ms"]}
    breakdown = [condition | {"stage": stage, "timing": "wall" if stage in WALL_STAGES else "cuda_event"}
                 | latency_summary([r[stage] for r in rows]) for stage in sorted(rows[0])]
    memory = condition | {"peak_allocated_gpu_mib": torch.cuda.max_memory_allocated() / 2**20,
                          "peak_reserved_gpu_mib": torch.cuda.max_memory_reserved() / 2**20,
                          "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}
    return {"throughput": timing, "latency": breakdown, "memory": memory}


def _bind(output: Path, inputs: Inputs, config: dict, keys: list, assets: dict) -> None:
    paths = [Path(__file__), *sorted((ROOT / "hla_tok/compression").glob("*.py")),
             ROOT / "hla_tok/streaming/compact.py", ROOT / "hla_tok/streaming/compact_wire.py"]
    record_path = output / "benchmark_manifest.json"
    record = inputs.provenance() | {"protocol": sha256_file(CONFIG), "benchmark": config["benchmark"],
                                   "assets": {key: assets[key]["sha256"] for key in keys}, "hardware": environment(),
                                   "representative_frames": [list(ref) for ref in representative(inputs.cohort)],
                                   "input": "GPU native RGB, file IO excluded", "output": "GPU native RGB",
                                   "external_network_wait": False,
                                   "implementation": {str(p.relative_to(ROOT)): sha256_file(p) for p in paths}}
    if record_path.exists() and json_at(record_path) != record:
        raise ValueError("benchmark resume identity differs")
    write_json(record_path, record)


def _publish(output: Path, cases: list[dict]) -> None:
    write_csv(output / "throughput_benchmark.csv", [case["throughput"] for case in cases])
    write_csv(output / "latency_breakdown.csv", [row for case in cases for row in case["latency"]])
    write_csv(output / "memory_benchmark.csv", [case["memory"] for case in cases])


def _validation_run(inputs: Inputs, config: dict, codec: CompactCodec, asset: dict, output: Path) -> None:
    cases = [(video, i) for video, n in inputs.cohort.items() for i in range(n)]
    sources = list(validation_frames(inputs, cases))
    rows: list[dict] = []
    for precision, budget in product(config["benchmark"]["precision"], budgets_for(config, asset)):
        path = TimedPath(inputs, codec, budget)
        try:
            with compute(precision):
                selected = _validation(path, sources)
        finally:
            path.close()
        rows.extend({"compute_dtype": precision, "budget": budget} | row for row in selected)
    write_csv(output / "validation_latencies_per_frame.csv", rows)
    from scripts.hla_tok.continuous.matrix import groups

    write_csv(output / "validation_latency.csv", [
        tags | {"frames": len(selected)} | latency_summary([row["total_serial"] for row in selected])
        for tags, selected in groups(rows, ("compute_dtype", "budget"))
    ])


def _deployment(output: Path, assets: dict, order: RankOrder, model_sha256: str) -> None:
    rows = []
    for name, spec in assets.items():
        before = torch.cuda.memory_allocated()
        codec, load = load_asset(output / "profiles" / spec["filename"], spec,
                                 PROFILES_BY_LABEL[spec["profile"]], order, model_sha256)
        allocated = torch.cuda.memory_allocated() - before
        rows.append({"asset": name, "profile": spec["profile"], "projection": spec["projection"],
                     "storage_bytes": spec["storage_bytes"], "asset_sha256": spec["sha256"],
                     "allocated_gpu_bytes_after_load": allocated} | load)
        del codec
    write_csv(output / "profile_deployment.csv", rows)


@torch.inference_mode()
def run(inputs: Inputs, config: dict, output: Path, selection: dict, assets: dict) -> None:
    keys = bench_assets(selection, assets)
    _bind(output, inputs, config, keys, assets)
    frames = [frame for _, _, frame in validation_frames(inputs, representative(inputs.cohort))]
    order = compact_orders()[selection["order"]]
    _deployment(output, assets, order, inputs.checkpoint["state_dict_sha256"])
    cases = []
    for key in keys:
        spec = assets[key]
        codec, _ = load_asset(output / "profiles" / spec["filename"], spec, PROFILES_BY_LABEL[spec["profile"]],
                              order, inputs.checkpoint["state_dict_sha256"])
        for precision, budget in product(config["benchmark"]["precision"], budgets_for(config, spec)):
            path = output / "benchmark_cases" / f"{key}_{precision}_{budget}.json"
            if path.exists():
                case = json_at(path)
            else:
                identity = {"asset": key, "profile": codec.profile.label, "order": order.name.name,
                            "compute_dtype": precision, "budget": budget}
                case = _measure(inputs, codec, identity, frames, config["benchmark"])
                write_json(path, case)
            cases.append(case)
            _publish(output, cases)
            data = case["throughput"]
            print(f"{path.stem}: {data['steady_state_fps']:.2f} FPS / P95 {data['p95_ms']:.3f} ms", flush=True)
        if key == selection["main_asset"]:
            _validation_run(inputs, config, codec, spec, output)


def main() -> None:
    parser = run_arguments(__doc__)
    parser.set_defaults(output=OUTPUT)
    args = parser.parse_args()
    with locked_output(args.output):
        inputs, config = compact_inputs(args.v3, args.output)
        assets = json_at(args.output / "compact_profile_manifest.json")
        selection = json_at(args.output / "no_training_selection.json")
        trained = args.output / "training_selection.json"
        if trained.is_file() and json_at(trained).get("adopted"):
            new = json_at(trained)
            selection["main_asset"] = new["asset"]
            assets[new["asset"]] = new["spec"]
        run(inputs, config, args.output, selection, assets)
        inputs.finish(args.output / "benchmark_inputs.json")


if __name__ == "__main__":
    main()
