"""Bounded 200-step, four-frame 1080p overfit before any long training."""

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.evaluation.gates import assess_smoke
from aip_vtok.evaluation.smoke import evaluate_smoke, write_curve
from aip_vtok.models.codec import create_codec
from aip_vtok.runtime import file_hash, load_config, resolve, save_checkpoint, write_json
from aip_vtok.training.budgets import sample_budgets
from aip_vtok.training.checkpoint import checkpoint_artifact, restore
from aip_vtok.training.checkpoint import trainable_state as trainable_state
from aip_vtok.training.data import read_window, source_by_id, sources
from aip_vtok.training.determinism import configure_determinism
from aip_vtok.training.diagnostics import gradient_diagnostics, loss_record
from aip_vtok.training.losses import ReconstructionLoss, crop_plan
from aip_vtok.training.trainer import RNGStreams
from cp_adaptok.metrics import MetricSuite


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
            loss, record = loss_record(
                objective,
                rgb,
                rec,
                crops,
                frame_id,
                budget,
                inspect_gradients and frame_id == 0 and index == 0,
            )
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


@dataclass
class SmokeRun:
    config: dict
    directory: Path
    checkpoint: Path
    codec: torch.nn.Module
    objective: torch.nn.Module
    suite: MetricSuite
    parameters: list
    optimizer: torch.optim.Optimizer
    rng: RNGStreams
    clip: torch.Tensor
    hashes: dict
    start: int
    history: list


def initialize_smoke(config, resume):
    gate = json.loads((resolve(config["output"]) / "gate0_backbone_and_ref.json").read_text())
    if not gate["passed"]:
        raise RuntimeError("Gate 0 is not released")
    directory = resolve(config["output"]) / "gate1_smoke"
    checkpoint = resolve(config["checkpoints"]) / "gate1_smoke/latest.pt"
    if checkpoint.exists() and not resume:
        raise FileExistsError("existing smoke checkpoint: use --resume")
    directory.mkdir(parents=True, exist_ok=True)
    configure_determinism()
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
    return SmokeRun(
        config,
        directory,
        checkpoint,
        codec,
        objective,
        suite,
        parameters,
        optimizer,
        rng,
        clip,
        hashes,
        start,
        history,
    )


def update_smoke(run, step):
    run.optimizer.zero_grad(set_to_none=True)
    loss, records = reconstruction_step(
        run.codec, run.clip, run.objective, run.rng, step % 50 == 0 or step == 1
    )
    gradients = gradient_diagnostics(run.codec)
    norm = torch.nn.utils.clip_grad_norm_(run.parameters, 1.0, error_if_nonfinite=True)
    run.optimizer.step()
    return {
        "step": step,
        "loss": loss,
        "module_gradients": gradients,
        "grad_norm": float(norm),
        "records": records,
        "rate_weight": 0.0,
    }


def record_smoke(run, step, started):
    config, history = run.config, run.history
    if step % 5 == 0 or step == 1:
        print(
            f"smoke step={step}/{config['stage1']['steps']} loss={history[-1]['loss']:.6f} "
            f"elapsed_s={time.perf_counter() - started:.1f}",
            flush=True,
        )
    if step % config["stage1"]["evaluate_every"] == 0:
        metrics = evaluate_smoke(config, run.codec, run.clip, run.suite, step)
        print(json.dumps({"step": step, "metrics": metrics["summary"]}), flush=True)
        save_checkpoint(
            run.checkpoint,
            checkpoint_artifact(step, run.codec, run.optimizer, run.rng, history, config, run.hashes),
        )
        write_json(
            run.directory / "training.json",
            {"history": history, "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20},
        )


def finish_smoke(run):
    config, history, directory = run.config, run.history, run.directory
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


def run(config, resume=False):
    session = initialize_smoke(config, resume)
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    for step in range(session.start + 1, config["stage1"]["steps"] + 1):
        session.history.append(update_smoke(session, step))
        record_smoke(session, step, started)
    return finish_smoke(session)


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
