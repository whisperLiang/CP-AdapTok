"""Bounded 200-step, four-frame 1080p overfit before any long training."""

import argparse
import json
import time

import torch

from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.evaluation.gates import assess_smoke
from aip_vtok.evaluation.smoke import evaluate_smoke, write_curve
from aip_vtok.models.codec import create_codec
from aip_vtok.runtime import file_hash, load_config, resolve, save_checkpoint, write_json
from aip_vtok.training.data import read_window, source_by_id, sources
from aip_vtok.training.losses import ReconstructionLoss, crop_plan
from aip_vtok.training.trainer import RNGStreams
from cp_adaptok.metrics import MetricSuite


def module_gradient_norm(module):
    gradients = [p.grad.detach().float().norm() for p in module.parameters() if p.grad is not None]
    return float(torch.stack(gradients).norm()) if gradients else 0.0


def gradient_diagnostics(codec):
    modules = {
        "anchor_encoder": codec.tokenizer.anchor,
        "innovation_encoder": codec.tokenizer.innovation,
        "detail_encoder": codec.tokenizer.detail,
        "temporal_predictor": codec.predictor,
        "fusion": codec.fusion,
    }
    modules.update({f"lift{index}": module for index, module in enumerate(codec.tokenizer.lifts)})
    return {name: module_gradient_norm(module) for name, module in modules.items()}


def sample_budgets(rng):
    anchor = 256 if torch.rand((), generator=rng.budget_rng) < 0.2 else 510
    middle = int(torch.randint(768, 2049, (), generator=rng.budget_rng))
    detail = int(torch.randint(3072, 8193, (), generator=rng.budget_rng))
    return [anchor, middle, detail]


def reconstruction_step(codec, clip, objective, rng, inspect_gradients=False):
    state, total_loss, records = ReferenceState(), 0.0, []
    initial_calls = codec.encode_calls
    for frame_id, frame in enumerate(clip):
        rgb = frame[None].cuda().float() / 255
        encoded = codec.encode_frame(rgb, state, frame_id)
        budgets = sample_budgets(rng)
        crops = crop_plan(1080, 1920, rng.crop_rng)
        losses = []
        for index, budget in enumerate(budgets):
            session = codec.prefix_session(encoded, state, budget, differentiable=True)
            rec = session.render_frame(checkpointed=True)
            loss, components = objective(rgb, rec, crops)
            record = {
                "frame": frame_id,
                "budget": budget,
                "components": {name: float(value.detach()) for name, value in components.items()},
            }
            if inspect_gradients and frame_id == 0 and index == 0:
                record["weighted_loss_gradient_norm_at_rgb"] = {
                    name: float(
                        torch.autograd.grad(objective.weights[name] * value, rec, retain_graph=True)[0].norm()
                    )
                    for name, value in components.items()
                }
            records.append(record)
            losses.append(loss)
        loss = torch.stack(losses).mean() / len(clip)
        loss.backward()
        total_loss += float(loss.detach())
        with torch.no_grad():
            budget = codec.reference_i if frame_id % codec.backbone.gop == 0 else codec.reference_p
            session = codec.prefix_session(encoded, state, budget)
            state = session.commit_reference_once()
    if codec.encode_calls - initial_calls != len(clip):
        raise RuntimeError("EncodeOnce contract violated")
    return total_loss, records


def trainable_state(codec):
    return {
        name: value
        for name, value in codec.state_dict().items()
        if not name.startswith(("backbone.", "projector."))
    }


def checkpoint_artifact(step, codec, optimizer, rng, history, config, hashes):
    return {
        "step": step,
        "model": trainable_state(codec),
        "optimizer": optimizer.state_dict(),
        "streams": rng.state_dict(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all(),
        "history": history,
        "config": config,
        "hashes": hashes,
    }


def restore(artifact, codec, optimizer, rng, config, hashes):
    if artifact["config"] != config or artifact["hashes"] != hashes:
        raise ValueError("resume config or source identity changed")
    result = codec.load_state_dict(artifact["model"], strict=False)
    if result.unexpected_keys or any(
        not name.startswith(("backbone.", "projector.")) for name in result.missing_keys
    ):
        raise ValueError("invalid trainable checkpoint tensors")
    optimizer.load_state_dict(artifact["optimizer"])
    rng.load_state_dict(artifact["streams"])
    torch.set_rng_state(artifact["torch_rng"])
    torch.cuda.set_rng_state_all(artifact["cuda_rng"])
    return artifact["step"], artifact["history"]


def run(config, resume=False):
    gate = json.loads((resolve(config["output"]) / "gate0_backbone_and_ref.json").read_text())
    if not gate["passed"]:
        raise RuntimeError("Gate 0 is not released")
    directory = resolve(config["output"]) / "gate1_smoke"
    checkpoint = resolve(config["checkpoints"]) / "gate1_smoke/latest.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError("existing smoke checkpoint: use --resume")
    directory.mkdir(parents=True, exist_ok=True)
    codec = create_codec(config).train()
    objective = ReconstructionLoss(config["stage1"]["loss"]).cuda()
    suite = MetricSuite("cuda:0")
    if suite.errors:
        raise RuntimeError(f"required metrics unavailable: {suite.errors}")
    parameters = [p for p in codec.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=config["stage1"]["lr"], weight_decay=0)
    rng = RNGStreams(config["seed"])
    rows, hashes = sources(config, "train")
    source = source_by_id(rows, config["stage1"]["video"])
    clip = read_window(source, config["stage1"]["start"], config["stage1"]["sequence_length"])
    hashes.update(
        {
            "alignment": file_hash(resolve(config["checkpoints"]) / "alignment.pt"),
            "projector": file_hash(resolve(config["checkpoints"]) / "projector_w128/latest.pt"),
        }
    )
    start, history = 0, []
    if resume:
        artifact = torch.load(checkpoint, map_location="cpu", weights_only=False)
        start, history = restore(artifact, codec, optimizer, rng, config, hashes)
    else:
        evaluate_smoke(config, codec, clip, suite, 0)
    write_json(
        directory / "inputs.json",
        {"config": config, "hashes": hashes, "source": source["content_id"], "sealed_test_opened": False},
    )
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(start + 1, config["stage1"]["steps"] + 1):
        optimizer.zero_grad(set_to_none=True)
        loss, records = reconstruction_step(codec, clip, objective, rng, step % 50 == 0 or step == 1)
        gradients = gradient_diagnostics(codec)
        norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
        optimizer.step()
        history.append(
            {
                "step": step,
                "loss": loss,
                "module_gradients": gradients,
                "grad_norm": float(norm),
                "records": records,
                "rate_weight": 0.0,
            }
        )
        if step % 5 == 0 or step == 1:
            print(
                f"smoke step={step}/{config['stage1']['steps']} loss={loss:.6f} "
                f"elapsed_s={time.perf_counter() - started:.1f}",
                flush=True,
            )
        if step % config["stage1"]["evaluate_every"] == 0:
            metrics = evaluate_smoke(config, codec, clip, suite, step)
            print(json.dumps({"step": step, "metrics": metrics["summary"]}), flush=True)
            save_checkpoint(
                checkpoint, checkpoint_artifact(step, codec, optimizer, rng, history, config, hashes)
            )
            write_json(
                directory / "training.json",
                {"history": history, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20},
            )
    final_step = config["stage1"]["steps"]
    first = json.loads((directory / "metrics_000000.json").read_text())
    final = json.loads((directory / f"metrics_{final_step:06d}.json").read_text())
    decision = assess_smoke(
        first["summary"],
        final["summary"],
        sum(r["loss"] for r in history[:10]) / 10,
        sum(r["loss"] for r in history[-10:]) / 10,
    )
    write_json(directory / "decision.json", decision)
    write_curve(directory)
    print(json.dumps(decision), flush=True)
    return decision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/aip_vtok/gate1_smoke.yaml")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    import yaml

    settings = yaml.safe_load(resolve(args.config).read_text())
    config = load_config(settings["gate0_config"])
    config["stage1"] = settings["stage1"]
    decision = run(config, args.resume)
    if not decision["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
