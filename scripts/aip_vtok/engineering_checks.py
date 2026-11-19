"""Small real-GPU IO/resume regressions in a separate diagnostic namespace."""

import argparse
import copy
import gc
import json
import shutil

import torch
import yaml

from aip_vtok.evaluation.evaluate import calibrate, create_backbone, load_alignment, run_baseline
from aip_vtok.evaluation.formal import evaluate
from aip_vtok.runtime import load_config, resolve, write_json
from aip_vtok.training.formal import settings, train
from aip_vtok.training.smoke import run as smoke
from aip_vtok.training.trainer import fit_projector


def prepare(namespace):
    config = settings("configs/aip_vtok/gate1.yaml", 7301)
    base_results, base_checkpoints = resolve(config["output"]), resolve(config["checkpoints"])
    directory, checkpoints = base_results / namespace, base_checkpoints / namespace
    directory.mkdir(parents=True, exist_ok=True)
    checkpoints.mkdir(parents=True, exist_ok=True)
    for name in ("alignment.pt", "projector_w128"):
        target, source = checkpoints / name, base_checkpoints / name
        if not target.exists():
            target.symlink_to(source, target_is_directory=source.is_dir())
        if target.resolve() != source.resolve():
            raise RuntimeError("engineering asset points to an unexpected source")
    for name in (
        "gate0_backbone_and_ref.json",
        "gate1_smoke/decision.json",
        "training_path_equivalence.json",
    ):
        destination = directory / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(base_results / name, destination)
    config["output"], config["checkpoints"] = str(directory), str(checkpoints)
    config["engineering_only"] = True
    config["stage1"].update(steps=1, save_every=1)
    config["stage1"]["validation"].update(max_frames_per_video=2, total_frames=6)
    return config


def release_cuda():
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()


def stage1_checks(config):
    checkpoint = resolve(config["checkpoints"]) / "gate1_seed7301/latest.pt"
    codec = train(config, resume=checkpoint.exists())
    del codec
    release_cuda()
    codec = train(config, resume=True)
    del codec
    release_cuda()
    result = evaluate(config)
    assert result["maximum_independent_rgb_error"] == 0
    assert result["summary"]["paired"]["8192"]["frames"] == 6
    release_cuda()
    smoke_config = copy.deepcopy(config)
    smoke_config["stage1"] = yaml.safe_load(resolve("configs/aip_vtok/gate1_smoke.yaml").read_text())[
        "stage1"
    ]
    smoke_config["stage1"].update(steps=2, evaluate_every=1)
    path = resolve(config["checkpoints"]) / "gate1_smoke/latest.pt"
    smoke(smoke_config, resume=path.exists())
    smoke(smoke_config, resume=True)
    release_cuda()
    return {
        "formal_update_and_resume": True,
        "six_frame_independent_evaluation": True,
        "bounded_smoke_update_and_resume": True,
    }


def gate0_checks(config):
    base = load_config("configs/aip_vtok/gate0.yaml")
    base["output"] = str(resolve(config["output"]) / "r0_reader_regression")
    base["checkpoints"] = config["checkpoints"]
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    old_cudnn = torch.backends.cudnn.deterministic
    try:
        # Compare the reader/state changes using the original R0 arithmetic policy.
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        repeated = run_baseline(base)
    finally:
        torch.use_deterministic_algorithms(old_deterministic)
        torch.backends.cudnn.deterministic = old_cudnn
    original = json.loads(resolve("results/aip_vtok/baseline.json").read_text())
    errors = {mode: abs(repeated[mode]["mse"] - original[mode]["mse"]) for mode in repeated}
    assert max(errors.values()) == 0, errors
    release_cuda()
    backbone = create_backbone(base)
    load_alignment(base, backbone)
    fitting = copy.deepcopy(base)
    fitting["output"] = str(resolve(config["output"]) / "projector_fit")
    fitting["projector_training"].update(steps=2, save_every=1, sequence_length=2)
    path = resolve(config["checkpoints"]) / "projector_w32/latest.pt"
    fit_projector(fitting, backbone, 32, resume=path.exists())
    fit_projector(fitting, backbone, 32, resume=True)
    calibration = copy.deepcopy(base)
    calibration["output"] = str(resolve(config["output"]) / "calibration")
    calibration["checkpoints"] = str(resolve(config["checkpoints"]) / "calibration")
    calibration["calibration"] = {"videos": ["DAVIS2017:bear"], "frames_per_video": 1}
    calibrate(calibration, backbone)
    return {
        "r0_mse_error_after_reader_and_state_fixes": errors,
        "projector_update_resume_and_calibration": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", choices=("engineering", "engineering_final"), default="engineering")
    args = parser.parse_args()
    config = prepare(args.namespace)
    result = {
        "scope": "engineering checks only; does not enter the formal dual-seed gate",
        "stage1": stage1_checks(config),
        "gate0": gate0_checks(config),
    }
    write_json(resolve(config["output"]) / "checks.json", result)
    print(result)


if __name__ == "__main__":
    main()
