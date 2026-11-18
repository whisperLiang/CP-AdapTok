"""Standalone native endpoint timing and isolated measured cross-scale overhead."""

import argparse
import json

import torch

from adaptok_hr.data.dataset import read_clip
from adaptok_hr.data.manifest import load_index
from adaptok_hr.runtime import environment, file_hash, seed_everything, write_json
from adaptok_hr.training.checkpoint import load_model
from cp_adaptok.metrics import summarize_latency
from experiments.adaptok_hr.fine_recovery.benchmark import require_idle_gpu, stream

from .run import ROOT, checkpoint


@torch.inference_mode()
def fusion_overhead(model, frame, repeats=100):
    fusion = model.encoder.cross_scale
    if fusion is None:
        return {"enabled": False, "parameters": 0}
    captures = []
    handle = fusion.register_forward_pre_hook(lambda module, args: captures.append(tuple(v.detach() for v in args[0])))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        model.encode_features(frame)
    handle.remove()
    features = captures[0]
    macs, handles = [], []
    def count(module, args, output):
        macs.append(output.numel() * module.in_channels // module.groups * module.kernel_size[0] * module.kernel_size[1])
    for module in fusion.modules():
        if isinstance(module, torch.nn.Conv2d):
            handles.append(module.register_forward_hook(count))
    with torch.autocast("cuda", dtype=torch.bfloat16):
        fusion(features)
    for handle in handles:
        handle.remove()
    rows, memory = fusion_latency(fusion, features, repeats)
    return {"enabled": True, "parameters": sum(p.numel() for p in fusion.parameters()),
            "input_shapes": [list(v.shape) for v in features], "conv_macs": sum(macs), "conv_flops": 2*sum(macs),
            "flops_scope": "convolutions only; bilinear, normalization, nonlinearities and additions excluded; latency includes all operations",
            "latency": summarize_latency(rows), "raw_latency_ms": rows, "repeats": repeats,
            "peak_incremental_gpu_mib": memory,
            "timing_scope": "isolated BF16 fusion on actual native encoder feature maps, CUDA events, 20 warmups, idle GPU"}



def fusion_latency(fusion, features, repeats):
    rows = []
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for _ in range(20):
            fusion(features)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(repeats):
            start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            fusion(features)
            stop.record()
            stop.synchronize()
            rows.append(start.elapsed_time(stop))
    return rows, (torch.cuda.max_memory_allocated() - baseline) / 2**20

def measure(name, step):
    path = checkpoint(name, step)
    output = ROOT / name / f"benchmark_{step}.json"
    if output.exists():
        if json.loads(output.read_text())["checkpoint_sha256"] != file_hash(path):
            raise ValueError("benchmark checkpoint identity mismatch")
        return
    require_idle_gpu()
    seed_everything(20261201)
    torch.set_num_threads(4)
    model, saved = load_model(path)
    index = load_index(saved["config"]["data"]["index"])
    source = next(r for r in index["splits"]["validation"] if r["content_id"] == "DAVIS2017:car-roundabout")
    clip, metadata = read_clip(source, 0, 64)
    budgets = {str(k): stream(model, clip, k) for k in model.config.budgets}
    fusion = fusion_overhead(model, clip[0:1].cuda())
    require_idle_gpu()
    write_json(output, {"environment": environment(), "checkpoint_sha256": file_hash(path),
               "completed_steps": saved["completed_steps"], "input": metadata, "budgets": budgets,
               "parameters": sum(p.numel() for p in model.parameters()), "fusion": fusion,
               "scope": "sequential native encoder+13-bit packing+sender local decode+independent receiver; synchronized wall; excludes input I/O,H2D,quality metrics and assertions",
               "idle_gpu_before_and_after": True})
    print(json.dumps({"name": name, "step": step, "joint_FPS": budgets["2048"]["timing"]["joint"]["fps"]}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", nargs="+", required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    for name in args.names:
        measure(name, args.steps)
