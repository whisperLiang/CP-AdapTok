"""Encode each batch once; decode all requested budgets from that exact full latent."""

import time

import torch

from .profiling import measured, synchronize
from .tiling import Stitcher, extract_batches


@torch.inference_mode()
def reconstruct_window(clip, adapter, layout, budgets, batch_size=4):
    if len(clip) != adapter.frame_num or layout.tile_size != adapter.input_size:
        raise ValueError("window/tile shape must follow the official checkpoint")
    synchronize(adapter.device)
    begin = time.perf_counter()
    source, transfer = measured(lambda: clip.to(adapter.device), adapter.device)
    writers = {k: Stitcher(layout, len(clip), 3, adapter.device) for k in budgets}
    timing = {"tile_batch_size": batch_size, "tile_count": len(layout.coordinates),
              "tiling_ms": transfer["wall_ms"], "encode_ms": 0., "encode_cuda_ms": 0.,
              "decode_ms_by_budget": {k: 0. for k in budgets}, "decode_cuda_ms_by_budget": {k: 0. for k in budgets},
              "stitch_ms_by_budget": {k: 0. for k in budgets}}
    batches, costs = measured(lambda: list(extract_batches(source, layout, batch_size)), adapter.device)
    timing["tiling_ms"] += costs["wall_ms"]
    for coordinates, tiles in batches:
        encoded, cost = measured(lambda: adapter.encode(tiles), adapter.device)
        timing["encode_ms"] += cost["wall_ms"]
        timing["encode_cuda_ms"] += cost["cuda_ms"] or 0.
        for budget, writer in writers.items():
            recon, cost = measured(lambda: adapter.decode(encoded, budget), adapter.device)
            timing["decode_ms_by_budget"][budget] += cost["wall_ms"]
            timing["decode_cuda_ms_by_budget"][budget] += cost["cuda_ms"] or 0.
            _, cost = measured(lambda: writer.add(coordinates, recon), adapter.device)
            timing["stitch_ms_by_budget"][budget] += cost["wall_ms"]
    results = {}
    for budget, writer in writers.items():
        results[budget], cost = measured(writer.finish, adapter.device)
        timing["stitch_ms_by_budget"][budget] += cost["wall_ms"]
    synchronize(adapter.device)
    timing["all_budgets_wall_ms"] = (time.perf_counter() - begin) * 1000
    # Account allocation, dispatch, and loop overhead for a complete standalone budget estimate.
    timed = timing["tiling_ms"] + timing["encode_ms"] + sum(timing["decode_ms_by_budget"].values()) + sum(timing["stitch_ms_by_budget"].values())
    timing["shared_overhead_ms"] = max(0., timing["all_budgets_wall_ms"] - timed)
    return results, timing
