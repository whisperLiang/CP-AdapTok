"""Verify complete packet equality and measure a matched eager/fused pipeline serially."""

import json

import torch

from cp_adaptok.backbones.causal_state import configure_fingerprint_backend
from cp_adaptok.data import read_video
from cp_adaptok.evaluation import benchmark_pipeline
from cp_adaptok.runtime import build_model, read_config, seed_everything, write_json
from scripts.measure_optimization import packet_trace


def main():
    root = "results/optimization_20261115"
    config = read_config(root + "/final_config.json")
    seed_everything(config["seed"], config["device"], 0.8)
    model, _, _ = build_model(config, config["device"], checkpoint_override=root + "/auxiliary/latest.pt")
    model.eval()
    clip = read_video(dict(config["data"]["test"][0], frames=32))
    with torch.no_grad():
        eager = packet_trace(model, clip)
        configure_fingerprint_backend("compiled")
        fused = packet_trace(model, clip)
    identical = all(a.packets == b.packets and a.header_bytes == b.header_bytes for a, b in zip(eager, fused))
    if not identical:
        raise RuntimeError("fused fingerprint changed exact state or packet bytes")
    reports = {}
    for backend in ["eager", "compiled"]:
        configure_fingerprint_backend(backend)
        reports[backend] = benchmark_pipeline(model, config, [1920, 1080], 32, 2, [128, 2048])
    write_json(
        root + "/fused_fingerprint_pipeline.json",
        {
            "checkpoint": root + "/auxiliary/latest.pt",
            "packet_identical_frames": 32,
            "bit_exact": True,
            "quality_gate_passed": False,
            "variants": reports,
            "warmup": 2,
            "cold_compile_excluded_from_warm_fps": True,
        },
    )
    print(json.dumps({k: v["overall"] for k, v in reports.items()}))


if __name__ == "__main__":
    main()
