"""Phase 2 entry point, guarded by the recorded tokenizer-readiness decision."""

import json
from pathlib import Path

import torch

from ..codec.codec import Codec, tensor_digest
from ..data.dataset import ClipSampler
from ..data.manifest import load_index
from ..models.scorer import BudgetScorer, scorer_update
from ..runtime import file_hash, seed_everything, write_json
from .checkpoint import load_model, reconcile_log, restore_rng, rng_state
from .train import append_row, execute_training, training_parser


def save_scorer(path, scorer, optimizer, scheduler, sampler, completed, identity):
    path = Path(path)
    state = {"scorer": scorer.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "sampling": sampler.state_dict(), "rng": rng_state(),
             "completed_steps": completed, "label_frames": completed, "read_frames": completed * 2,
             "frozen_identity": identity, "budgets": scorer.budgets}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)
    write_json(path.with_suffix(".json"), {"completed_steps": completed, "label_frames": completed,
                                         "read_frames": completed * 2, "frozen_identity": identity,
                                         "checkpoint_sha256": file_hash(path)})


def restore_scorer(path, scorer, optimizer, scheduler, sampler, identity):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state["frozen_identity"] != identity or tuple(state["budgets"]) != scorer.budgets:
        raise ValueError("scorer frozen tokenizer/data identity mismatch")
    scorer.load_state_dict(state["scorer"], strict=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    sampler.load_state_dict(state["sampling"])
    restore_rng(state["rng"])
    return state["completed_steps"]


def train_scorer(args):
    readiness = json.loads(Path(args.readiness).read_text())
    if not readiness.get("scorer_ready", False):
        raise ValueError("tokenizer readiness decision does not permit Phase 2")
    seed_everything(20261117)
    torch.set_num_threads(4)
    model, _ = load_model(args.checkpoint, args.device)
    model.eval().requires_grad_(False)
    initial = tensor_digest(model.state_dict()).hex()
    index = load_index(args.index)
    sampler = ClipSampler(index["splits"]["train"], 20261118)
    scorer = BudgetScorer(16, model.config.budgets).to(args.device).train()
    optimizer = torch.optim.AdamW(scorer.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    identity = {"checkpoint": file_hash(args.checkpoint), "manifest": index["identity"]}
    output = Path(args.output)
    completed = restore_scorer(args.resume, scorer, optimizer, scheduler, sampler, identity) if args.resume else 0
    if not args.resume and (output / "training.jsonl").exists():
        raise ValueError("existing scorer run requires --resume")
    reconcile_log(output / "training.jsonl", completed)
    codec = Codec(model, amp_bf16=False)
    while completed < args.steps:
        clip, metadata = sampler.sample(2, model.config.height, model.config.width)
        _, state = codec.encode_frame(clip[0:1].to(args.device), None, model.config.budgets[-1], metadata["content_id"])
        measured = scorer_update(model, scorer, optimizer, clip[1:2].to(args.device), state)
        scheduler.step()
        completed += 1
        append_row(output / "training.jsonl", {"step": completed, **measured, "input": metadata})
        if completed % 100 == 0 or completed == args.steps:
            save_scorer(output / f"step_{completed:06d}.pt", scorer, optimizer, scheduler, sampler, completed, identity)
    unchanged = tensor_digest(model.state_dict()).hex() == initial
    if not unchanged:
        raise RuntimeError("Phase 2 changed frozen tokenizer weights")
    write_json(output / "completion.json", {"completed_steps": completed, "tokenizer_unchanged": unchanged,
                                            "label_source": "real decoded ID losses at the same P-frame causal state"})


def main():
    parser = training_parser(__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--readiness", required=True)
    parser.add_argument("--index", default="data/adaptok_hr/index.json")
    parser.add_argument("--device", default="cuda")
    execute_training(train_scorer, parser.parse_args())


if __name__ == "__main__":
    main()
