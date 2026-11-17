"""Joint VQ training with real reconstruction history, nested auxiliaries, and TBPTT."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from ..config import ModelConfig
from ..data.dataset import ClipSampler, read_clip
from ..data.manifest import load_index
from ..models.model import AdapTokHR
from ..models.temporal_state import frame_context
from ..models.vq import code_statistics
from ..runtime import environment, read_config, seed_everything, stable_hash, write_json
from .checkpoint import reconcile_log, restore_training, save_training
from .losses import ReconstructionLoss


def frame_objective(model, frame, state, previous_target, budget, auxiliary, losses):
    encoded = model.encode_features(frame, state)
    main = model.forward_frame(frame, state, budget, encoded)
    loss, terms = losses.distortion(main.rgb, frame)
    _, _, reference = frame_context(state, model.config.gop)
    temporal = losses.temporal(main.rgb, frame, reference, previous_target)
    mono = loss.new_zeros(())
    if auxiliary is not None:
        alternative = model.forward_frame(frame, state, auxiliary, encoded)
        extra, other = losses.distortion(alternative.rgb, frame)
        loss = (loss + extra) * .5
        low, high = (terms["mse"], other["mse"]) if budget < auxiliary else (other["mse"], terms["mse"])
        mono = F.relu(high - low) * losses.weights.get("monotonic", 0)
    total = loss + temporal + mono + losses.weights.get("vq", 0) * encoded.loss
    row = {k: float(v.detach()) for k, v in terms.items()}
    row.update(vq_loss=float(encoded.loss.detach()), temporal_loss=float(temporal.detach()),
               monotonic_loss=float(mono.detach()), main_budget=budget, auxiliary_budget=auxiliary,
               frame_type=main.frame_type, frame_index=main.state.frame_index,
               **code_statistics(main.ids), quantization_mse=encoded.stats["quantization_mse"])
    return total, main.state, row


def module_change_norms(model, before):
    result = {}
    for group in ("encoder", "tokenizer", "vq", "decoder"):
        squares = [(p.detach() - before[n]).float().square().sum()
                   for n, p in model.named_parameters() if n.startswith(group + ".")]
        result[group] = float(torch.stack(squares).sum().sqrt())
    return result


class JointTrainer:
    def __init__(self, model, settings):
        if settings["tbptt"] < 1 or settings["nested_every"] < 1:
            raise ValueError("positive TBPTT and nested interval required")
        if settings["budget_mode"] not in ("fixed", "nested"):
            raise ValueError("budget_mode must be fixed or nested")
        self.model, self.settings = model, settings
        self.device = next(model.parameters()).device
        self.losses = ReconstructionLoss(settings["losses"], self.device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=settings["learning_rate"],
                                          weight_decay=settings["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1.0)

    def budgets(self, step):
        if self.settings["budget_mode"] == "fixed":
            return self.settings["fixed_budget"], None
        budgets = self.model.config.budgets
        main = budgets[int(torch.randint(len(budgets), ()))]
        other = None
        if step % self.settings["nested_every"] == 0:
            candidates = [b for b in budgets if b != main]
            other = candidates[int(torch.randint(len(candidates), ()))]
        return main, other

    def backward_clip(self, clip, step):
        state, previous, pending, rows = None, None, [], []
        for index, frame in enumerate(clip):
            budget, auxiliary = self.budgets(step)
            with torch.autocast(self.device.type, dtype=torch.bfloat16,
                                enabled=self.settings["amp_bf16"] and self.device.type == "cuda"):
                loss, state, row = frame_objective(self.model, frame[None], state, previous,
                                                   budget, auxiliary, self.losses)
            pending.append(loss / len(clip))
            rows.append(row)
            previous = frame[None]
            if (index + 1) % self.settings["tbptt"] == 0 or index + 1 == len(clip):
                torch.stack(pending).sum().backward()
                state, pending = state.detach(), []
        return rows

    def update(self, clip, step):
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        before = {n: p.detach().clone() for n, p in self.model.named_parameters()} if step == 1 or step % 100 == 0 else None
        if self.device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        rows = self.backward_clip(clip.to(self.device), step)
        norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.settings["gradient_clip"],
                                             error_if_nonfinite=True)
        self.optimizer.step()
        self.scheduler.step()
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        mse = sum(r["mse"] for r in rows) / len(rows)
        result = {"step": step, "frames": len(rows), "mse": mse, "psnr_global": -10 * math.log10(max(mse, 1e-20)),
                  "gradient_norm": float(norm), "learning_rate": self.optimizer.param_groups[0]["lr"],
                  "update_seconds": time.perf_counter() - started,
                  "peak_gpu_mib": torch.cuda.max_memory_allocated() / 2**20 if self.device.type == "cuda" else 0,
                  "budgets": [r["main_budget"] for r in rows], "frame_metrics": rows}
        if before is not None:
            result["module_update_norms"] = module_change_norms(self.model, before)
        return result


def sequence_length(schedule, completed):
    return max((s for s in schedule if completed >= s[0]), key=lambda s: s[0])[1]


def append_row(path, row):
    with Path(path).open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def prepare_run(args, config, index, trainer, sampler):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "final_test_seal.json").exists():
        raise ValueError("final-test-sealed run cannot resume training")
    completed, frames = 0, 0
    if args.resume:
        state = restore_training(args.resume, trainer, sampler, config, index["identity"])
        completed, frames = state["completed_steps"], state["training_frames"]
    elif (output / "run.json").exists():
        raise ValueError("existing run requires explicit --resume")
    discarded = reconcile_log(output / "training.jsonl", completed)
    write_json(output / "run.json", {"environment": environment(), "config": config,
                                     "manifest_identity": index["identity"], "requested_steps": args.steps,
                                     "resume_from": args.resume, "starting_completed_steps": completed,
                                     "discarded_uncommitted_log_rows": discarded,
                                     "experiment_type": "single_train_clip_fit" if args.overfit else "joint_native_1080p"})
    return output, completed, frames


def run_training(args):
    config = read_config(args.config)
    config["run_scope"] = {"overfit": args.overfit, "overfit_frames": args.overfit_frames}
    seed_everything(config["training"]["seed"])
    torch.set_num_threads(4)
    index = load_index(config["data"]["index"])
    records = index["splits"]["train"]
    if args.overfit:
        records = [r for r in records if r["content_id"] == args.overfit]
        if not records:
            raise ValueError("overfit must use a Train source")
    model = AdapTokHR(ModelConfig(**config["model"])).to(args.device)
    trainer = JointTrainer(model, config["training"])
    sampler = ClipSampler(records, config["training"]["seed"] + 1)
    output, completed, frames = prepare_run(args, config, index, trainer, sampler)
    if completed == 0:
        save_training(Path(args.checkpoints) / output.name / "step_000000.pt", trainer, sampler,
                      config, index["identity"], 0, 0)
        if not args.overfit:
            from ..evaluation.evaluate import validate_fast

            write_json(output / "validation_000000.json", validate_fast(
                model, index, config["evaluation"]["validation_ids"], frames=8,
                amp_bf16=config["evaluation"]["amp_bf16"]))
    cached = read_clip(records[0], 0, args.overfit_frames, model.config.height, model.config.width) if args.overfit else None
    while completed < args.steps:
        length = sequence_length(config["training"]["sequence_schedule"], completed)
        clip, metadata = cached if cached is not None else sampler.sample(length, model.config.height, model.config.width)
        row = trainer.update(clip, completed + 1)
        completed, frames = completed + 1, frames + len(clip)
        row.update(training_frames=frames, input=metadata)
        append_row(output / "training.jsonl", row)
        if completed == 1 or completed % 25 == 0:
            print(json.dumps({k: row[k] for k in ("step", "training_frames", "psnr_global", "gradient_norm",
                                                  "update_seconds", "peak_gpu_mib")}), flush=True)
        if completed % config["training"]["checkpoint_every"] == 0 or completed == args.steps:
            path = Path(args.checkpoints) / Path(args.output).name / f"step_{completed:06d}.pt"
            metadata_ckpt = save_training(path, trainer, sampler, config, index["identity"], completed, frames)
            write_json(output / "latest_checkpoint.json", {"path": str(path.resolve()), **metadata_ckpt})
        if completed % config["training"]["validation_every"] == 0 and not args.overfit:
            from ..evaluation.evaluate import validate_fast

            validation = validate_fast(model, index, config["evaluation"]["validation_ids"],
                                       frames=8, amp_bf16=config["evaluation"]["amp_bf16"])
            write_json(output / f"validation_{completed:06d}.json", validation)
    write_json(output / "completion.json", {"completed_steps": completed, "training_frames": frames,
                                           "target_steps": args.steps, "complete": completed == args.steps,
                                           "config_identity": stable_hash(config), "manifest_identity": index["identity"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/adaptok_hr/default.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--checkpoints", default="checkpoints/adaptok_hr")
    parser.add_argument("--overfit", help="Train content ID; independent single-clip fit diagnostic")
    parser.add_argument("--overfit-frames", type=int, default=4)
    run_training(parser.parse_args())


if __name__ == "__main__":
    main()
