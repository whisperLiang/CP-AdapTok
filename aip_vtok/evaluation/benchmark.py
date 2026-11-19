"""CUDA-event latency of the direct-reference projector (not codec FPS)."""

import torch

from cp_adaptok.metrics import summarize_latency


@torch.no_grad()
def projector_latency(projector, repeats=100, warmup=20):
    latent = torch.randn(1, 128, 68, 120, device="cuda").to(memory_format=torch.channels_last)
    result = {}
    for intra, label in ((True, "I"), (False, "P")):
        for _ in range(warmup):
            projector(latent, intra)
        torch.cuda.synchronize()
        events = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(repeats)
        ]
        for start, end in events:
            start.record()
            projector(latent, intra)
            end.record()
        torch.cuda.synchronize()
        result[label] = summarize_latency([start.elapsed_time(end) for start, end in events])
    result["maximum_mean_ms"] = max(result[k]["mean_ms"] for k in ("I", "P"))
    result["measurement"] = "FP32 eager channels_last; CUDA events; includes both output heads separately"
    result["codec_fps"] = None
    return result
