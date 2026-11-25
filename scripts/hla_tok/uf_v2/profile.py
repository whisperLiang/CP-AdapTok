"""Measure the pinned UF-LD, UF-Anchor, and D0 before any optimizations."""
import argparse
from collections import defaultdict
from collections.abc import Callable
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from statistics import mean
from time import perf_counter
from unittest.mock import patch

import torch
from torch.profiler import ProfilerActivity, profile, record_function

import hla_tok.uf.anchor as anchor
from hla_tok.evaluation.io import write_json
from hla_tok.uf.base import schedule
from hla_tok.uf.entropy import SegmentCoder
from hla_tok.uf.latent import UnifiedCodec, UnifiedRefinement
from hla_tok.uf.official import NativeDecoder, NativeEncoder, NativeNAL, OfficialAssets
from hla_tok.uf.refinement import RefinementQuantizer
from hla_tok.uf.wire import PrefixFamily
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.anchor_eval import AnchorEncoder, locked_qp
from scripts.hla_tok.uf.inputs import OUTPUT, inputs
from scripts.hla_tok.uf.unified_eval import UnifiedEncoder
from scripts.hla_tok.v2.protocol import setup


def decorate(owner, name: str, times: dict, stack: ExitStack, label: str) -> None:
    original = getattr(owner, name)
    @wraps(original)
    def measured(*args, **kwargs):
        begin = perf_counter()
        with record_function(label):
            result = original(*args, **kwargs)
        times[label].append((perf_counter() - begin) * 1000)
        return result
    stack.enter_context(patch.object(owner, name, measured))


def run_case(method: str, assets: OfficialAssets, qp: tuple[int, int], checkpoint: Path,
             frames, out: Path) -> dict:
    native = NativeDecoder(assets)
    if method == "ld":
        enc = NativeEncoder(assets)
        writer, parser = NativeNAL(), NativeNAL()

        def evaluate(rgb, index):
            intra, reset = schedule(index)
            base = enc.encode(rgb, index, qp[0 if intra else 1], intra, reset)
            raw = writer.encode(base)
            native.decode(parser.decode(raw, index))
    else:
        model = UnifiedRefinement().cuda().eval() if method == "d0" else anchor.AnchorModel().cuda().eval()
        if method == "anchor":
            model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False)["model"])
            enc = AnchorEncoder(assets, anchor.RefinementCodec(model, "bf16"), qp)
        else:
            enc = UnifiedEncoder(assets, UnifiedCodec(model), qp)

        def evaluate(rgb, index):
            intra, reset = schedule(index)
            result = enc.encode(rgb, index, intra, reset)
            raw, _ = result.family.prefix(20833)
            state = enc.receiver(native)
            state.feed(raw)
            state.decode_available()
            state.finish()
            state.render()
    def frame(evaluate: Callable, rgb, index: int) -> float:
        torch.cuda.synchronize()
        begin = perf_counter()
        with record_function("application.frame"):
            evaluate(rgb, index)
            torch.cuda.synchronize()
        return (perf_counter() - begin) * 1000

    for i, rgb in enumerate(frames[:4]):
        frame(evaluate, rgb, i)
    wall = [frame(evaluate, rgb, i) for i, rgb in enumerate(frames[4:8], 4)]
    calls = [(NativeEncoder, "encode", "native.encode"), (NativeDecoder, "decode", "native.decode"),
             (RefinementQuantizer, "symbols", "enh.quantize_and_copy"),
             (anchor.RecordCodec, "_scores", "enh.group_and_score"),
             (SegmentCoder, "encode", "enh.rans_encode"), (SegmentCoder, "decode", "enh.rans_decode"),
             (anchor, "priority_order", "enh.priority_iterator_creation"),
             (anchor.IntegerCanvas, "apply", "enh.apply"),
             (PrefixFamily, "build", "enh.build_inclusive"),
             (anchor.FrameReceiver, "commit_base", "base.commit_inclusive"),
             (anchor.FrameReceiver, "render", "display.render"),
             (anchor.AnchorModel, "analyze", "enh.anchor_analysis")]
    times: dict = defaultdict(list)
    with ExitStack() as stack:
        for owner, name, label in calls:
            decorate(owner, name, times, stack, label)
        instrumented = [frame(evaluate, rgb, i) for i, rgb in enumerate(frames[8:12], 8)]
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], profile_memory=True) as trace:
        traced = [frame(evaluate, rgb, i) for i, rgb in enumerate(frames[12:], 12)]
    trace_path = out / f"{method}.trace.json"
    trace.export_chrome_trace(str(trace_path))
    averages = trace.key_averages()
    return {"steady_p_wall_ms": wall, "instrumented_wall_ms": instrumented, "trace_wall_ms": traced,
            "trace": trace_path.name, "phase_frames": len(instrumented),
            "phase": {name: {"calls": len(vals), "ms_per_frame": sum(vals) / len(instrumented)}
                      for name, vals in times.items()},
            "cuda_sync_calls": sum(item.count for item in averages if "Synchronize" in item.key),
            "steady_p_fps": 1000 / mean(wall)}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup()
    args.trace_dir.mkdir(parents=True, exist_ok=True)
    reader, plans = inputs()
    frames = [source_frame(reader, plans["validation"][0], i) for i in range(14)]
    qp = locked_qp(OUTPUT / "baseline/isolated/base_lock.json")
    assets = OfficialAssets.verify(args.weights)
    results = {}
    for name in ("ld", "anchor", "d0"):
        results[name] = run_case(name, assets, qp, args.anchor, frames, args.trace_dir)
        print(name, results[name]["steady_p_fps"], flush=True)
    write_json(args.output, {"profile": results, "read_counts": reader.counts, "qp": qp})


if __name__ == "__main__":
    main()
