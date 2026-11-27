"""Measure both native AdapTok tiling modes on the shared Validation clip."""

import argparse
import json
from hashlib import sha256
from pathlib import Path
from time import perf_counter

import cv2
import torch

from baselines.adaptok.cli import warmup
from baselines.adaptok.metrics import FullFrameMetrics, boundary_mask
from baselines.adaptok.model_adapter import OfficialAdapter
from baselines.adaptok.reconstruction import reconstruct_window
from baselines.adaptok.tiling import TileLayout
from baselines.adaptok.video_reader import read_input, temporal_windows
from common.data import dataset_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("validation", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    d = json.loads(args.validation.read_text())
    assert d["split"] == "validation"
    item, = [r for r in d["records"] if r["content_id"] == "DAVIS2017:car-roundabout"]
    path = dataset_path(item["path"])
    assert item["role"] == "validation" and "validation" in path.resolve().parts
    source, metadata = read_input(path, frames=17)
    assert source.shape == (17, 3, 1080, 1920)
    begin = perf_counter()
    adapter = OfficialAdapter(args.checkpoint)
    torch.cuda.synchronize()
    load_ms = (perf_counter() - begin) * 1000
    metric = FullFrameMetrics(adapter.device)
    mask = boundary_mask(1080, 1920, device=adapter.device)
    checks, timing = {}, {}
    for stride in (128, 96):
        layout = TileLayout(1080, 1920, stride=stride)
        warmup(adapter, metadata, {"tile_size": 128, "tile_batch_size": 4, "warmup_batches": 3,
                                  "budgets": [512, 2048], "strides": [stride]})
        torch.cuda.reset_peak_memory_stats()
        windows = list(temporal_windows(source, adapter.frame_num))
        values = []
        for _ in range(3):
            outputs, phases = reconstruct_window(windows[0].frames, adapter, layout, [512, 2048])
            values.append(phases)
        tail, _ = reconstruct_window(windows[1].frames, adapter, layout, [512, 2048])
        outputs = {k: torch.cat((v, tail[k][:1])) for k, v in outputs.items()}
        timing[stride] = {"repetitions": values, "peak_bytes": torch.cuda.max_memory_allocated()}
        checks[stride] = {k: {"shape": list(v.shape), "sha256": sha256(v.cpu().contiguous().numpy()).hexdigest(),
                              "quality": [metric(a[None].cuda(), b[None], mask) for a,b in zip(source, v, strict=True)],
                              "tokens": k * len(layout.coordinates) * len(windows), "serialized_bytes": None}
                          for k,v in outputs.items()}
        del outputs
    args.output.write_text(json.dumps({"input": {"content": item["content_id"], "frame_hashes": item["frame_hashes"][:17],
                                               "test_pixel_reads": 0}, "checks": checks, "timing": timing,
                                       "load_ms": load_ms, "model": adapter.verify_unchanged()["state_sha256_after"],
                                       "gpu": torch.cuda.get_device_name(), "torch": torch.__version__},
                                       indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
