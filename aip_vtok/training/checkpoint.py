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
    artifact = {
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
    if "model_version" in config:
        artifact["model_version"] = codec.model_version
        if config["joint_projector"]:
            artifact["projector"] = codec.projector.state_dict()
    return artifact


def load_fix_weights(codec, artifact):
    if artifact.get("model_version") != codec.model_version:
        raise ValueError("Gate 1-Fix model version mismatch; historical checkpoints require the legacy codec")
    joint = artifact["config"]["joint_projector"]
    if joint != ("projector" in artifact):
        raise ValueError("Gate 1-Fix projector weights do not match the declared training policy")
    load_trainable(codec, artifact["model"])
    if joint:
        codec.projector.load_state_dict(artifact["projector"])


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
    if "model_version" in config:
        load_fix_weights(codec, artifact)
    else:
        load_trainable(codec, artifact["model"])
    restore_training_state(artifact, optimizer, rng)
    return artifact["step"], artifact["history"]
