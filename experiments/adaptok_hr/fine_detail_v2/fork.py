"""Create an explicitly identified sibling at a committed optimizer boundary."""

import copy
import json

import torch

from adaptok_hr.codec.codec import tensor_digest
from adaptok_hr.config import ModelConfig
from adaptok_hr.data.dataset import ClipSampler
from adaptok_hr.data.manifest import load_index
from adaptok_hr.models.model import AdapTokHR
from adaptok_hr.runtime import file_hash, stable_hash, write_json
from adaptok_hr.training.checkpoint import restore_rng, rng_state, save_training
from adaptok_hr.training.train import JointTrainer, exclusive_run

from .run import ROOT, checkpoint


def named_moments(names, optimizer):
    group, = optimizer["param_groups"]
    return {f"{name}/{key}": tensor for name, pid in zip(names, group["params"], strict=True)
            for key, tensor in optimizer["state"].get(pid, {}).items()}


def adopt_training_state(parent_model, saved, child, allowed_new):
    common = saved["model"]
    missing = child.model.load_state_dict(common, strict=False)
    if missing.unexpected_keys or any(not key.startswith(allowed_new) for key in missing.missing_keys):
        raise ValueError(f"unapproved child/parent differences: {missing}")
    before = tensor_digest(common).hex()
    after = tensor_digest({k: child.model.state_dict()[k] for k in common}).hex()
    if before != after:
        raise ValueError("inherited weight tensors changed")
    original = saved["optimizer"]
    old_group, = original["param_groups"]
    current = child.optimizer.state_dict()
    group, = current["param_groups"]
    parent_names = list(dict(parent_model.named_parameters()))
    child_names = list(dict(child.model.named_parameters()))
    old_ids = dict(zip(parent_names, old_group["params"], strict=True))
    new_ids = dict(zip(child_names, group["params"], strict=True))
    states = {new_ids[name]: copy.deepcopy(original["state"][pid])
              for name, pid in old_ids.items() if pid in original["state"]}
    new_group = {**copy.deepcopy(old_group), "params": group["params"]}
    child.optimizer.load_state_dict({"state": states, "param_groups": [new_group]})
    child.scheduler.load_state_dict(copy.deepcopy(saved["scheduler"]))
    old_digest = tensor_digest(named_moments(parent_names, original)).hex()
    new_digest = tensor_digest(named_moments(child_names, child.optimizer.state_dict())).hex()
    if old_digest != new_digest:
        raise ValueError("inherited optimizer moments changed")
    return {"parent_model_tensors": before, "child_inherited_model_tensors": after,
            "parent_optimizer_tensors": old_digest, "child_inherited_optimizer_tensors": new_digest,
            "additional_parameters": sorted(set(child_names) - set(parent_names))}


def fork_at(parent, child, config, step=1000):
    source = checkpoint(parent, step)
    target = checkpoint(child, step)
    output = ROOT / child
    expected = {"parent": parent, "parent_checkpoint_sha256": file_hash(source),
                "child": child, "child_config_sha256": stable_hash(config), "step": step}
    with exclusive_run(output):
        marker = output / "fork.json"
        if marker.exists():
            previous = json.loads(marker.read_text())
            if (previous["lineage"] != expected or not target.exists()
                    or previous["child_checkpoint_sha256"] != file_hash(target)):
                raise ValueError("fork identity mismatch")
            return target
        if (output / "run.json").exists():
            raise ValueError("training output already exists without a verified fork")
        saved = torch.load(source, map_location="cpu", weights_only=False)
        if (saved["completed_steps"] != step or saved["config_identity"] != stable_hash(saved["config"])
                or expected["parent_checkpoint_sha256"] != json.loads(source.with_suffix(".json").read_text())["checkpoint_sha256"]):
            raise ValueError("invalid committed parent")
        index = load_index(config["data"]["index"])
        if saved["manifest_identity"] != index["identity"]:
            raise ValueError("fork changes input manifest")
        # Resume needs the complete, independent log preceding this boundary.
        with (ROOT / parent / "training.jsonl").open() as old:
            lines = [line for line in old if json.loads(line)["step"] <= step]
        if [json.loads(line)["step"] for line in lines] != list(range(1, step + 1)):
            raise ValueError("parent log has a missing optimizer step")
        pending = output / "training.jsonl.tmp"
        pending.write_text("".join(lines))
        pending.replace(output / "training.jsonl")
        outer = rng_state()
        try:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(config["training"]["model_seed"])
                parent_model = AdapTokHR(ModelConfig(**saved["config"]["model"]))
                child_model = AdapTokHR(ModelConfig(**config["model"]))
            trainer = JointTrainer(child_model, config["training"])
            proof = adopt_training_state(parent_model, saved, trainer, ("decoder.fine_detail.",))
            sampler = ClipSampler(index["splits"]["train"], 0)
            sampler.load_state_dict(saved["sampling"])
            trainer.budget_rng.set_state(saved["budget_rng"])
            restore_rng(saved["rng"])
            meta = save_training(target, trainer, sampler, config, index["identity"], step, saved["training_frames"])
        finally:
            restore_rng(outer)
        write_json(marker, {"lineage": expected, "child_checkpoint_sha256": meta["checkpoint_sha256"],
                            "inherited_log_rows": step, **proof})
    return target
