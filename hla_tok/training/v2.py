"""Matched initialization and strict architecture identity for V2 checkpoints."""

from hashlib import sha256

import torch

from hla_tok import HLATok
from hla_tok.models.anchors import anchor_digest
from hla_tok.training.checkpoint import read_checkpoint

SHARED = ("encoder.", "query_encoder.fine.", "query_encoder.fuse.", "query_encoder.position",
          "local_decoder.", "rgb_decoder.", "quantizer.")


def state_digest(state, names):
    digest = sha256()
    for name in sorted(names):
        digest.update(name.encode())
        digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def matched_initialization(arm, seed):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        reference = HLATok("R0")
        torch.manual_seed(seed)
        candidate = HLATok(arm)
    source = reference.state_dict()
    target = candidate.state_dict()
    copied = [name for name in target if name in source and target[name].shape == source[name].shape]
    target.update({name: source[name] for name in copied})
    candidate.load_state_dict(target, strict=True)
    shared = [name for name in copied if name.startswith(SHARED)]
    metadata = {
        "seed": seed, "architecture": candidate.architecture_metadata(),
        "shared_sha256": state_digest(target, shared),
        "parameter_count": sum(p.numel() for p in candidate.parameters()),
        "initialization": "torch seed; matching parameters copied from one fresh R0; new layers use default seeded init",
    }
    return candidate, metadata


def load_experimental_checkpoint(path, model, dataset_hash):
    checkpoint = read_checkpoint(path)
    state = checkpoint.state
    if state["identity"]["dataset_sha256"] != dataset_hash:
        raise ValueError("checkpoint dataset manifest differs")
    expected = model.architecture_metadata()
    if state["counters"].get("architecture") != expected:
        raise ValueError("checkpoint architecture identity differs")
    table = state["model"].get("query_encoder.anchor_table")
    if table is not None and anchor_digest(table) != expected["anchor_table_sha256"]:
        raise ValueError("checkpoint anchor buffer differs from the deterministic table")
    model.load_state_dict(state["model"], strict=True)
    return checkpoint
