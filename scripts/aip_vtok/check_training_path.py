"""Real-GPU loss/gradient equivalence and timing, without changing saved weights."""

import time

import torch
import yaml

from aip_vtok.models.codec import create_codec
from aip_vtok.runtime import load_config, resolve, write_json
from aip_vtok.training.data import read_window, source_by_id, sources
from aip_vtok.training.losses import ReconstructionLoss
from aip_vtok.training.paired_step import paired_step
from aip_vtok.training.smoke import reconstruction_step
from aip_vtok.training.trainer import RNGStreams


def measure(operation, codec, clip, objective):
    codec.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    started = time.perf_counter()
    loss, _ = operation(codec, clip, objective, RNGStreams(111))
    torch.cuda.synchronize()
    return {
        "loss": loss,
        "wall_s": time.perf_counter() - started,
        "gradients": {
            name: p.grad.detach().clone() for name, p in codec.named_parameters() if p.grad is not None
        },
    }


def main():
    config = load_config("configs/aip_vtok/gate0.yaml")
    settings = yaml.safe_load(resolve("configs/aip_vtok/gate1_smoke.yaml").read_text())
    config["stage1"] = settings["stage1"]
    codec = create_codec(config).train()
    artifact = torch.load(
        resolve(config["checkpoints"]) / "gate1_smoke/latest.pt", map_location="cpu", weights_only=False
    )
    codec.load_state_dict(artifact["model"], strict=False)
    rows, _ = sources(config, "train")
    clip = read_window(source_by_id(rows, config["stage1"]["video"]), 0, 4)
    objective = ReconstructionLoss(config["stage1"]["loss"]).cuda()
    original = measure(reconstruction_step, codec, clip, objective)
    bulk = measure(paired_step, codec, clip, objective)
    error = max(
        float((original["gradients"][name] - value).abs().max()) for name, value in bulk["gradients"].items()
    )
    result = {
        "original_loss": original["loss"],
        "shared_context_loss": bulk["loss"],
        "loss_absolute_error": abs(original["loss"] - bulk["loss"]),
        "maximum_parameter_gradient_error": error,
        "original_step_s": original["wall_s"],
        "shared_context_step_s": bulk["wall_s"],
        "passed": abs(original["loss"] - bulk["loss"]) < 1e-7 and error < 1e-6,
        "codec_fps": None,
        "note": "training diagnostic only, four frames, no optimizer update",
    }
    with torch.autocast("cuda", dtype=torch.bfloat16):
        amp = measure(paired_step, codec, clip, objective)
        warm_amp = [measure(paired_step, codec, clip, objective) for _ in range(2)]
    result["training_bf16"] = {
        "loss": amp["loss"],
        "step_s": amp["wall_s"],
        "warm_step_s": [row["wall_s"] for row in warm_amp],
        "all_gradients_finite": all(torch.isfinite(g).all().item() for g in amp["gradients"].values()),
        "validation_precision": "FP32; BF16 used only for training activations",
    }
    write_json("results/aip_vtok/training_path_equivalence.json", result)
    print(result)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
