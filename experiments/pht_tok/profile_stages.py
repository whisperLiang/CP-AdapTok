"""Measure stage boundaries separately from the uninstrumented throughput run."""

import argparse
import ctypes
import json
from collections import defaultdict
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from statistics import median
from time import perf_counter
from unittest.mock import patch

import torch

import pht_tok.codec
from experiments.pht_tok.study import ROOT, SKIP, uvg_source, write_json
from hla_tok.hts.native import NativeNetworks
from pht_tok.bitstream import Chunk
from pht_tok.codec import PHTDecoder, PHTEncoder, model_identifier
from pht_tok.entropy import NativeEntropy
from pht_tok.official_adapter import HTSAdapter
from third_party.DCVC.src.utils.common import set_torch_env


class StageTimers:
    def __init__(self):
        self.times = defaultdict(list)

    def wrap(self, name, call):
        @wraps(call)
        def measure(*args, **kwargs):
            torch.cuda.synchronize()
            start = perf_counter()
            result = call(*args, **kwargs)
            torch.cuda.synchronize()
            self.times[name].append((perf_counter() - start) * 1000)
            return result
        return measure

    def wrap_methods(self, stack, obj, methods):
        for attr, label in methods.items():
            stack.enter_context(patch.object(obj, attr, self.wrap(label, getattr(obj, attr))))


class TimedProxy:
    def __init__(self, proxy, timers, side):
        self.proxy = proxy
        self.timers = timers
        self.side = side

    def __getattr__(self, name):
        value = getattr(self.proxy, name)
        labels = {"pht_prepare": "reference preparation", "pht_analyze": "shared HT-S CUDA analysis graph",
                  "pht_prior": "hyperprior and entropy index generation", "pht_group": "spatial latent accumulation",
                  "pht_render": "synthesis including missing-mean completion", "pht_display": "display head only"}
        return self.timers.wrap(f"{self.side} {labels[name]}", value) if name in labels else value


def feed(tx, rx, source, i):
    raw = tx.encode_chunk_once(source.yuv[i:i + 8], 32, i, 2)
    session = rx.decode_prefix(raw.data)
    session.reconstruct()
    return session


def api_probe(tx, rx, source):
    runtime = ctypes.CDLL(None)
    try:
        start = runtime.pht_count_start
        stop = runtime.pht_count_stop
    except AttributeError:
        return {"status": "UNAVAILABLE", "reason": "CUDA runtime counter library was not preloaded"}
    names = ("cudaDeviceSynchronize", "cudaStreamSynchronize", "cudaEventSynchronize",
             "cudaMemcpy", "cudaMemcpyAsync", "cudaGraphLaunch")
    counts = (ctypes.c_uint64 * len(names))()
    start.restype = stop.restype = None
    torch.cuda.synchronize()
    start()
    feed(tx, rx, source, 49)
    torch.cuda.synchronize()
    stop(counts)
    return dict(zip(names, counts, strict=True)) | {"includes_one_final_measurement_device_sync": True}


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--uvg", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=ROOT / ".build/pht-tok")
    args = parser.parse_args()
    set_torch_env()
    with torch.cuda.stream(torch.cuda.Stream()):
        source = uvg_source(args.uvg, 57)
        ident = model_identifier(args.weights, SKIP)
        timers = StageTimers()
        tx = PHTEncoder(NativeNetworks(args.weights, "hts", SKIP), ident)
        rx = PHTDecoder(NativeNetworks(args.weights, "hts", SKIP), ident)
        entropy_setup = timers.wrap("CDF table setup and entropy objects", NativeEntropy)(tx.adapter.model)
        del entropy_setup
        rx.decode_prefix(tx.encode_intra(source.yuv[:1], 32, 0).data)
        for i in (1, 9, 17):
            feed(tx, rx, source, i)
        original = HTSAdapter.proxy.fget
        wrapped = {id(endpoint.adapter): TimedProxy(original(endpoint.adapter), timers, side)
                   for endpoint, side in ((tx, "Encoder"), (rx, "Decoder"))}
        with ExitStack() as stack:
            stack.enter_context(patch.object(HTSAdapter, "proxy", property(lambda obj: wrapped[id(obj)])))
            timers.wrap_methods(stack, pht_tok.codec, {"assemble": "chunk assembly"})
            timers.wrap_methods(stack, Chunk, {"pack": "serialization checksums and record headers"})
            timers.wrap_methods(stack, tx.entropy, {"start_mask_encoding": "entropy worker submission and packing",
                                                   "finish_mask_encoding": "entropy worker collection"})
            timers.wrap_methods(stack, rx.entropy, {"decode_y": "signed Y rANS and record termination",
                                                   "decode_z": "Z rANS and hyperprior device copy"})
            timers.wrap_methods(stack, tx, {"commit": "Encoder reference commit"})
            timers.wrap_methods(stack, rx, {"commit": "Decoder reference commit"})
            for i in (25, 33, 41):
                feed(tx, rx, source, i)
        counts = api_probe(tx, rx, source)
    path = args.work / "profiling.json"
    prior = json.loads(path.read_text())
    prior["stage_diagnostics"] = {
        "scope": "three isolated, GPU-synchronized full 1080p runs at QP 32, P1 R1 prefix 2",
        "aggregation": "split boundaries introduce extra synchronization. Encoder/hyper/quantization share one graph",
        "per_call_ms": {name: {"median": median(times), "raw": times} for name, times in timers.times.items()},
        "cuda_api_counts_one_full_chunk": counts,
    }
    write_json(path, prior)


if __name__ == "__main__":
    main()
