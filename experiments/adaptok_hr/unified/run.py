"""Matched experiments orchestrate the existing JointTrainer; no alternate training loop."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch
import yaml

from adaptok_hr.config import ModelConfig
from adaptok_hr.data.dataset import ClipSampler
from adaptok_hr.data.manifest import load_index
from adaptok_hr.models.model import AdapTokHR
from adaptok_hr.runtime import file_hash, read_config, stable_hash, write_json
from adaptok_hr.training.checkpoint import restore_rng, rng_state, save_training
from adaptok_hr.training.train import JointTrainer, exclusive_run
from experiments.adaptok_hr.fine_detail_v2.fork import adopt_training_state
from experiments.adaptok_hr.fine_detail_v2.run import execute

PACKAGE = Path(__file__).resolve().parent
ROOT = Path("results/adaptok_hr_unified")
CHECKPOINTS = Path("checkpoints/adaptok_hr_unified")
DEFAULT_PARENTS = Path("/home/whisperliang/CP-AdapTok")
ARMS = ("U0", "U1", "U2", "U3")
ARCHITECTURES = ("N0", "N1", "N2")
SCRATCH_STEPS = (300, 1000, 3000, 5000)


def checkpoint(name, step):
    return CHECKPOINTS / name / f"step_{step:06d}.pt"


def parent_path(parent_root, seed):
    return parent_root / "checkpoints/adaptok_hr_fine_detail_v2" / f"S{seed}_D3/step_001300.pt"


def configuration(parent_root, seed, arm):
    source = parent_path(parent_root, seed)
    config = copy.deepcopy(read_config(source.with_suffix(".json"))["config"])
    calibration = json.loads((ROOT / "calibration.json").read_text())
    weights = calibration["chosen_weights"]
    mode, net = ("legacy", "alex") if arm == "U0" else ("unified", "alex" if arm == "U1" else "vgg")
    settings = config["training"]
    settings.update(budget_mode="paired", paired_probability=.75 if arm == "U3" else .5,
                    extra_checkpoints=[], milestones=[1600])
    settings["reconstruction"] = {"mode": mode, "network": net, "global_size": [270, 480],
                                   "crop_size": 256, "crop_seed": 20261201 + (seed - 1) * 10}
    if arm != "U0":
        settings["losses"] = {"l1": 1., "perceptual": weights[net], "vq": 1.}
    config["experiment"] = {"family": "unified", "stage": "A", "seed": seed, "arm": arm,
                            "name": f"S{seed}_{arm}", "initialization": "D3_step_1300_fork",
                            "parent_checkpoint_sha256": file_hash(source),
                            "calibration_sha256": file_hash(ROOT / "calibration.json")}
    return config


def write_config(name, config):
    path = ROOT / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    content = yaml.safe_dump(config, sort_keys=False)
    if path.exists() and read_config(path) != config:
        raise ValueError("existing experiment configuration differs")
    path.write_text(content)
    return path


def fork(parent_root, seed, arm, config):
    source, name = parent_path(parent_root, seed), f"S{seed}_{arm}"
    output, target = ROOT / name, checkpoint(name, 1300)
    lineage = {"parent_checkpoint_sha256": file_hash(source), "child_config_identity": stable_hash(config),
               "step": 1300, "seed": seed, "arm": arm}
    with exclusive_run(output):
        marker = output / "fork.json"
        if marker.exists():
            proof = json.loads(marker.read_text())
            if proof["lineage"] != lineage or proof["checkpoint_sha256"] != file_hash(target):
                raise ValueError("existing fork identity differs")
            return
        saved, index, lines = validated_parent(source, parent_root, seed, config)
        outer = rng_state()
        try:
            trainer = JointTrainer(AdapTokHR(ModelConfig(**config["model"])), config["training"])
            proof = adopt_training_state(trainer.model, saved, trainer, ())
            sampler = ClipSampler(index["splits"]["train"], 0)
            sampler.load_state_dict(saved["sampling"])
            trainer.budget_rng.set_state(saved["budget_rng"])
            restore_rng(saved["rng"])
            metadata = save_training(target, trainer, sampler, config, index["identity"], 1300, saved["training_frames"])
        finally:
            restore_rng(outer)
        (output / "training.jsonl").write_text("\n".join(lines) + "\n")
        write_json(marker, {"lineage": lineage, "checkpoint_sha256": metadata["checkpoint_sha256"], **proof})



def validated_parent(source, parent_root, seed, config):
    saved = torch.load(source, map_location="cpu", weights_only=False)
    parent_meta = json.loads(source.with_suffix(".json").read_text())
    if (saved["completed_steps"] != 1300 or parent_meta["checkpoint_sha256"] != file_hash(source)
            or saved["config_identity"] != stable_hash(saved["config"])):
        raise ValueError("invalid D3 parent checkpoint")
    index = load_index(config["data"]["index"])
    if saved["manifest_identity"] != index["identity"]:
        raise ValueError("fork changed data identity")
    log = parent_root / "results/adaptok_hr_fine_detail_v2" / f"S{seed}_D3/training.jsonl"
    lines = [s for s in log.read_text().splitlines() if json.loads(s)["step"] <= 1300]
    if [json.loads(s)["step"] for s in lines] != list(range(1, 1301)):
        raise ValueError("parent optimizer log is incomplete")
    return saved, index, lines

def train(name, config, steps):
    path = write_config(name, config)
    final = checkpoint(name, steps)
    completion = ROOT / name / "completion.json"
    if completion.exists() and json.loads(completion.read_text())["completed_steps"] >= steps and final.exists():
        return
    command = ["adaptok_hr.training.train", "--config", path, "--output", ROOT / name,
               "--checkpoints", CHECKPOINTS, "--steps", steps]
    committed = sorted(p for p in final.parent.glob("step_*.pt") if p.with_suffix(".json").exists())
    if committed:
        command += ["--resume", committed[-1]]
    started = time.perf_counter()
    try:
        execute(command, ROOT / "logs" / f"{name}_train_{steps}.log")
    finally:
        write_json(ROOT / name / f"timing_{steps}.json", {
            "trainer_process_wall_seconds": time.perf_counter() - started,
            "target_step": steps, "scope": "trainer process including initialization, I/O, validation, checkpoints; experiment fork excluded"})


def scratch_configuration(parent_root, seed, arm, loss_arm):
    config = configuration(parent_root, seed, loss_arm)
    config["model"].update(fine_input="direct" if arm == "N2" else "local_residual",
                           encoder_fusion="none" if arm == "N0" else "cross_scale_v1")
    config["training"].update(milestones=list(SCRATCH_STEPS),
                              sequence_schedule=[[0, 4], [1000, 16], [3000, 32]])
    config["experiment"].update(stage="B", arm=arm, name=f"S{seed}_{arm}", initialization="from_scratch",
                                loss_arm=loss_arm)
    del config["experiment"]["parent_checkpoint_sha256"]
    return config



def previous_milestone(step):
    index = SCRATCH_STEPS.index(step)
    if index == 0:
        raise ValueError("the first scratch milestone has no predecessor")
    return SCRATCH_STEPS[index - 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("pilot", "scratch"))
    parser.add_argument("--parent-root", type=Path, default=DEFAULT_PARENTS)
    parser.add_argument("--seeds", nargs="+", type=int, choices=(1, 2), default=[1, 2])
    parser.add_argument("--arms", nargs="+")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--loss-arm", choices=("U1", "U2", "U3"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    for seed in args.seeds:
        for arm in args.arms or (ARMS if args.stage == "pilot" else ARCHITECTURES):
            run_arm(args, seed, arm)


def run_arm(args, seed, arm):
    if args.stage == "pilot":
        if arm not in ARMS:
            raise ValueError("unknown pilot arm")
        config = configuration(args.parent_root, seed, arm)
        write_config(f"S{seed}_{arm}", config)
        fork(args.parent_root, seed, arm, config)
        train(f"S{seed}_{arm}", config, 1600)
    else:
        check_scratch_request(args, arm)
        config = scratch_configuration(args.parent_root, seed, arm, args.loss_arm)
        train(f"S{seed}_{arm}", config, args.steps)


def check_scratch_request(args, arm):
    if arm not in ARCHITECTURES or args.steps not in SCRATCH_STEPS or args.loss_arm is None:
        raise ValueError("scratch requires architecture, declared loss, and milestone")
    decision = json.loads((ROOT / "decision_A.json").read_text())
    if decision["winner"] != args.loss_arm:
        raise ValueError("scratch loss must be selected by the two-seed Pilot")
    if args.steps > 300:
        gate = json.loads((ROOT / f"decision_B_{previous_milestone(args.steps)}.json").read_text())
        if not gate["advance"] or arm not in gate["continuation_arms"]:
            raise ValueError("scratch quality/stability gate failed for this architecture")


if __name__ == "__main__":
    main()
