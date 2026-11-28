"""CUDA-synchronized wall measurements and latency percentiles."""

from collections.abc import Iterable
from time import perf_counter
from typing import Any

import numpy as np
import torch


def timed(call, *args: Any):
    torch.cuda.synchronize()
    begin = perf_counter()
    result = call(*args)
    torch.cuda.synchronize()
    return result, (perf_counter() - begin) * 1000


def timed_cuda(call):
    start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    torch.cuda.synchronize()
    begin = perf_counter()
    start.record()
    result = call()
    stop.record()
    stop.synchronize()
    return result, (perf_counter() - begin) * 1000, start.elapsed_time(stop)


def distribution(values: Iterable[float], label: str) -> dict:
    array = np.asarray(list(values))
    return {f"{label}_mean": float(array.mean()), f"{label}_max": float(array.max())} | {
        f"{label}_p{pct}": float(np.percentile(array, pct)) for pct in (50, 95, 99)}
