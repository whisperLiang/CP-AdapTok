"""Recover trainable tensors and RNG only; frozen weights cannot be overwritten."""

import torch


def trainable_state(codec):
    return {
        name: value
        for name, value in codec.state_dict().items()
        if not name.startswith(("backbone.", "projector."))
    }


def load_trainable(codec, state):
    if state.keys() != trainable_state(codec).keys():
        raise ValueError("invalid trainable checkpoint tensors; frozen weights are forbidden")
    codec.load_state_dict(state, strict=False)


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


def require_identity(artifact, config, hashes, hash_key="hashes"):
    if artifact["config"] != config or artifact[hash_key] != hashes:
        raise ValueError("resume config or source identity changed")


def restore_training_state(artifact, optimizer, rng, stream_key="streams"):
    optimizer.load_state_dict(artifact["optimizer"])
    rng.load_state_dict(artifact[stream_key])
    torch.set_rng_state(artifact["torch_rng"])
    torch.cuda.set_rng_state_all(artifact["cuda_rng"])


def restore(artifact, codec, optimizer, rng, config, hashes):
    require_identity(artifact, config, hashes)
    load_trainable(codec, artifact["model"])
    restore_training_state(artifact, optimizer, rng)
    return artifact["step"], artifact["history"]
