"""Opt-in synchronized wall/CUDA measurements, kept separate from throughput runs."""

import contextlib
import time

import torch


class PipelineProfiler:
    def __init__(self, device):
        self.device, self.records = torch.device(device), []

    @contextlib.contextmanager
    def measure(self, name):
        gpu = self.device.type == "cuda"
        if gpu:
            torch.cuda.synchronize(self.device)
            start_event, end_event = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start_event.record()
        start = time.perf_counter()
        yield
        if gpu:
            end_event.record()
            end_event.synchronize()
        self.records.append(
            {
                "name": name,
                "wall_ms": (time.perf_counter() - start) * 1000,
                "cuda_ms": start_event.elapsed_time(end_event) if gpu else None,
            }
        )


@contextlib.contextmanager
def measure(model, name):
    profiler = getattr(model, "pipeline_profiler", None)
    if profiler is None:
        yield
    else:
        with profiler.measure(name):
            yield
