"""Independent checkpointable random streams and an update-based LR schedule."""

import hashlib
import json
import math

import torch


class TrainingRNGStreams:
    names = ("data_rng", "augmentation_rng", "budget_rng", "quantization_rng")

    def __init__(self, seed, device="cpu"):
        for name, offset in zip(self.names, (0, 1009, 2027, 3041)):
            target = device if name == "quantization_rng" else "cpu"
            setattr(self, name, torch.Generator(device=target).manual_seed(seed + offset))

    def items(self):
        return [(name, getattr(self, name)) for name in self.names]

    def state_dict(self):
        return {name: rng.get_state() for name, rng in self.items()}

    def load_state_dict(self, state):
        if set(state) != set(self.names):
            raise ValueError("checkpoint must contain all four sampling RNG streams")
        for name, rng in self.items():
            rng.set_state(state[name])

    def get_state(self):
        # Legacy field remains readable by older structural tools.
        return self.data_rng.get_state()

    def set_state(self, state):
        self.data_rng.set_state(state)


def lr_factor(completed, warmup, total, minimum):
    """LR for the next update. The first warmup update is nonzero."""
    if completed < warmup:
        return (completed + 1) / max(1, warmup)
    progress = min(1.0, (completed - warmup) / max(1, total - warmup))
    return minimum + (1 - minimum) * (1 + math.cos(math.pi * progress)) / 2


def make_scheduler(optimizer, training):
    settings = training.get("scheduler")
    if not settings:
        return None
    warmup, total = settings["warmup_steps"], settings["total_steps"]
    minimum = settings.get("min_lr_ratio", 0.1)
    if not 0 <= warmup < total or not 0 <= minimum <= 1:
        raise ValueError("invalid warmup/cosine schedule")
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, warmup, total, minimum)
    )


def stable_hash(value):
    # YAML permits integer budget keys; JSON checkpoints/config snapshots stringify them.
    normalized = json.loads(json.dumps(value))
    return hashlib.sha256(json.dumps(normalized, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
