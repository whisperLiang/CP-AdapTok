"""Serial actual-bitstream measurements on identical weights; no competing task subprocesses."""

import json
from dataclasses import replace

import torch

from cp_adaptok.data import read_video
from cp_adaptok.evaluation import benchmark_pipeline
from cp_adaptok.models.causal_progressive_codec import CausalProgressiveCodec
from cp_adaptok.runtime import build_model, metadata, read_config, seed_everything, write_json
from cp_adaptok.streaming.sender import CodecEncoder

ROOT = "results/optimization_20261115"


def packet_trace(model, clip):
    encoder = CodecEncoder(model)
    return [encoder.encode_frame(frame[None].to(model.device)) for frame in clip]


def main():
    config = read_config(ROOT + "/final_config.json")
    checkpoint = ROOT + "/auxiliary/latest.pt"
    seed_everything(config["seed"], config["device"], 0.8)
    model, _, _ = build_model(config, config["device"], checkpoint_override=checkpoint)
    model.eval()
    results = {
        "metadata": metadata(config, "quality_gate_failed_engineering_only"),
        "checkpoint": checkpoint,
        "source_io_excluded": True,
        "quality_gate_passed": False,
        "variants": {},
    }
    for name, budgets in [("reference_five", [128, 256, 512, 1024, 2048]), ("reference_two", [128, 2048])]:
        results["variants"][name] = benchmark_pipeline(model, config, [1920, 1080], 8, 2, budgets)
    clip = read_video(dict(config["data"]["test"][0], frames=8))
    with torch.no_grad():
        expected = packet_trace(model, clip)
        model.vq.chunk_size = 8192
        actual = packet_trace(model, clip)
    matches = all(
        a.packets == b.packets and a.header_bytes == b.header_bytes for a, b in zip(expected, actual)
    )
    results["vq_large_batch_packet_identical"] = matches
    if not matches:
        raise RuntimeError("larger VQ batch changed deterministic bitstream; reject optimization")
    results["variants"]["vq_large_batch_two"] = benchmark_pipeline(
        model, config, [1920, 1080], 8, 2, [128, 2048]
    )
    candidate = CausalProgressiveCodec(replace(model.config, attention_precision="float32"))
    candidate.load_state_dict(model.state_dict(), strict=True)
    candidate.to(model.device).eval()
    results["variants"]["fp32_experimental_two"] = benchmark_pipeline(
        candidate, config, [1920, 1080], 8, 2, [128, 2048]
    )
    write_json(ROOT + "/performance_comparison.json", results)
    print(json.dumps({k: v["overall"] for k, v in results["variants"].items()}))


if __name__ == "__main__":
    main()
