"""Large-scale experiment lifecycle, with immutable inputs and exact phase semantics."""

import copy
import importlib.util
import json
import subprocess
from dataclasses import asdict
from datetime import datetime
from zoneinfo import ZoneInfo

import torch

from cp_adaptok.large_scale_data import file_hash, load_manifests, summarize_split, video_specs
from cp_adaptok.large_scale_tracking import TrainingObserver, read_json, reconcile_training_logs
from cp_adaptok.models.causal_progressive_codec import CodecConfig
from cp_adaptok.runtime import (
    MissingPrerequisites,
    build_model,
    metadata,
    preflight,
    read_config,
    resolve,
    seed_everything,
    write_json,
)
from cp_adaptok.training import run_training
from cp_adaptok.training_state import stable_hash

PHASES = ("shared_warmup", "s1", "s2", "s3", "s4")


def gpu_snapshot():
    command = ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
               "--format=csv,noheader,nounits"]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    fields = result.stdout.strip().splitlines()[0].split(",")
    index, name, total, used, free, utilization = [value.strip() for value in fields]
    processes = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                                "--format=csv,noheader"], capture_output=True, text=True, check=True)
    return {"index": int(index), "name": name, "total_mib": int(total), "used_mib": int(used),
            "free_mib": int(free), "utilization_percent": int(utilization),
            "compute_processes": processes.stdout.strip(), "exclusive_gpu_assumed": False}


def memory_fraction(snapshot, settings):
    if snapshot["free_mib"] < settings["gpu_min_free_mib"]:
        raise MissingPrerequisites(["insufficient currently free GPU memory; no other process was terminated"])
    safe = (snapshot["free_mib"] - settings["gpu_reserve_mib"]) / snapshot["total_mib"]
    return round(min(0.8, safe), 3)


def enforce_architecture(config):
    actual = CodecConfig(**config["model"])
    required = {"n_max": 2048, "k_ref": 128, "packet_tokens": 128, "codebook_size": 8192,
                "embedding_dim": 16, "token_aggregation": "global_softmax_v1", "query_content_only": False}
    for name, value in required.items():
        if getattr(actual, name) != value:
            raise ValueError(f"large-scale experiment fixes {name}={value}")
    return stable_hash(asdict(actual))


def scale_conditions(splits, settings, crop=None):
    summary = summarize_split(splits["train"])
    conditions = []
    if summary["independent_videos"] < settings["minimum_train_videos"]:
        conditions.append(f"only {summary['independent_videos']} independent training videos")
    if summary["frames"] < settings["minimum_train_frames"]:
        conditions.append(f"only {summary['frames']} actual training frames")
    native = summary["eligible_videos_by_crop"]["1080x1920"]
    if native < settings["minimum_native1080_videos"]:
        conditions.append(f"only {native} native >=1080p training videos")
    if crop is not None and not video_specs(splits["train"], crop=crop):
        conditions.append(f"no native training sources fit {crop}")
    return conditions


def run_preflight():
    config = read_config("configs/large_scale/shared_warmup.yaml")
    splits, hashes = load_manifests(config["large_scale"]["manifest_dir"])
    snapshot = gpu_snapshot()
    config["gpu_memory_fraction"] = memory_fraction(snapshot, config["large_scale"])
    config["data"] = {split: video_specs(rows) for split, rows in splits.items()}
    preflight(config, "train")
    from cp_adaptok.metrics import MetricSuite

    metrics = MetricSuite("cpu", ["lpips", "dists", "ms-ssim"])
    report = metadata(config, "preflight_complete")
    report.update(gpu=snapshot, selected_memory_fraction=config["gpu_memory_fraction"], manifest_hashes=hashes,
                  datasets={split: summarize_split(rows) for split, rows in splits.items()},
                  scale_blockers=scale_conditions(splits, config["large_scale"]),
                  metric_errors=metrics.errors, native_extensions={name: importlib.util.find_spec(name).origin
                      for name in ("MLCodec_extensions_cpp", "inference_extensions_cuda")},
                  weights={name: {"path": config["paths"][name], "sha256": file_hash(resolve(config["paths"][name]))}
                           for name in ("ld_checkpoint", "intra_checkpoint")})
    write_json("results/large_scale/preflight.json", report)
    print(json.dumps({key: report[key] for key in ("gpu", "selected_memory_fraction", "scale_blockers",
                                                  "metric_errors")}, indent=2), flush=True)
    return report


def experiment(run_id=None, seed=None):
    common = read_config("configs/large_scale/common.yaml")
    root = resolve(common["large_scale"]["output_root"])
    current = read_json(root / "active_run.json", {})
    run_id = experiment_id(run_id, current)
    directory = root / run_id
    saved = directory / "experiment_manifest.json"
    if saved.exists():
        manifest = read_json(saved, {})
        if seed is not None and seed != manifest["seed"]:
            raise ValueError("existing experiment has a different seed; choose a new run-id")
    else:
        manifest = new_experiment(directory, common, run_id, seed)
        write_json(saved, manifest)
    splits, hashes = load_manifests(manifest["manifest_directory"])
    if hashes != manifest["manifest_hashes"]:
        raise ValueError("experiment dataset manifests changed")
    write_json(root / "active_run.json", {"run_id": run_id})
    write_json(root / "experiment_manifest.json", manifest)
    return manifest, splits


def experiment_id(requested, current):
    value = requested or current.get("run_id") or datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%dT%H%M%S")
    if not value or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for c in value):
        raise ValueError("run-id must contain only ASCII letters, numbers, underscores, or hyphens")
    return value


def read_experiment(run_id=None):
    """Readers never rewrite the training writer's active pointer or mutable manifest."""
    common = read_config("configs/large_scale/common.yaml")
    root = resolve(common["large_scale"]["output_root"])
    run_id = experiment_id(run_id, read_json(root / "active_run.json", {}))
    path = root / run_id / "experiment_manifest.json"
    if not path.exists():
        raise MissingPrerequisites([f"existing experiment {run_id}"])
    manifest = read_json(path, {})
    splits, hashes = load_manifests(manifest["manifest_directory"])
    if hashes != manifest["manifest_hashes"]:
        raise ValueError("experiment dataset manifests changed")
    return manifest, splits


def new_experiment(directory, common, run_id, seed):
    splits, hashes = load_manifests(common["large_scale"]["manifest_dir"])
    snapshot_dir = directory / "manifests"
    for split in splits:
        source = resolve(common["large_scale"]["manifest_dir"]) / f"cpat_{split}.json"
        write_json(snapshot_dir / source.name, json.loads(source.read_text()))
    manifest = metadata(status="prepared")
    manifest.update(run_id=run_id, seed=seed if seed is not None else common["seed"], phases={},
                    manifest_hashes=hashes, manifest_directory=str(snapshot_dir),
                    dataset_summary={split: summarize_split(rows) for split, rows in splits.items()},
                    planned_updates={"shared_warmup": 2000, **{p: v["updates"] for p, v in
                                        common["large_scale"]["phases"].items()}},
                    fairness="equal optimizer updates and data windows; never claimed equal FLOPs",
                    final_test_frozen=False, single_seed=True, convergence_verified=False,
                    cuda_resume_numerics="RNG/data replay exact; adaptive_avg_pool2d_backward_cuda is not bitwise deterministic")
    return manifest


def source_checkpoint(phase, branch, manifest):
    if phase in ("pilot", "shared_warmup"):
        return None
    previous = PHASES[PHASES.index(phase) - 1]
    previous_branch = "shared" if previous == "shared_warmup" else branch
    common = read_config("configs/large_scale/common.yaml")["large_scale"]
    source = resolve(common["checkpoint_root"]) / manifest["run_id"] / previous / previous_branch / "best_validation.pt"
    state = manifest["phases"].get(f"{previous}/{previous_branch}", {})
    if state.get("status") != "complete" or not source.is_file():
        raise MissingPrerequisites([f"completed {previous}/{previous_branch} and its best_validation checkpoint"])
    return source


def phase_config(phase, branch, manifest, splits):
    filename = "shared_warmup" if phase in ("pilot", "shared_warmup") else "baseline" if branch == "B0" else "nested"
    config = copy.deepcopy(read_config(f"configs/large_scale/{filename}.yaml"))
    config["seed"] = manifest["seed"]
    settings = config["large_scale"]
    train = config["training"]
    target = apply_phase_settings(config, phase)
    crop = train["crop"]
    config["data"] = {"train": video_specs(splits["train"], crop=crop),
                      "validation": video_specs(splits["validation"]), "test": video_specs(splits["test"])}
    run_id = manifest["run_id"]
    output = resolve(settings["output_root"]) / run_id / phase / branch
    checkpoints = resolve(settings["checkpoint_root"]) / run_id / phase / branch
    source = source_checkpoint(phase, branch, manifest)
    config["paths"].update(output=str(output), save_checkpoint=str(checkpoints / "latest.pt"),
                            checkpoint=str(source) if source else None)
    config["experiment"] = {"run_id": run_id, "phase": phase, "branch": branch, "target_updates": target,
                            "manifest_hashes": manifest["manifest_hashes"],
                            "structure_hash": enforce_architecture(config),
                            "initial_checkpoint_sha256": file_hash(source) if source else None,
                            "official_weights_sha256": {k: file_hash(resolve(config["paths"][k]))
                                                         for k in ("ld_checkpoint", "intra_checkpoint")}}
    return config


def apply_phase_settings(config, phase):
    train, settings = config["training"], config["large_scale"]
    target = 2000
    if phase == "pilot":
        target = 200
        # Real pretrained-backbone optimization, including the VQ transition, in a separate disposable phase.
        train.update(save_every=50, validate_every=100, full_validate_every=1000000, diagnostic_every=50)
        train["scheduler"].update(total_steps=200, warmup_steps=20)
        train["vq"].update(warmup_steps=20, anneal_steps=200)
        config["teacher"].update(decay_steps=150)
    elif phase != "shared_warmup":
        stage = dict(settings["phases"][phase])
        target = stage.pop("updates")
        for name in ("lpips", "estimated_rate"):
            if name in stage:
                config["loss"][name] = stage.pop(name)
        train.update(stage)
        train["scheduler"].update(total_steps=target, warmup_steps=min(200, target // 10))
    return target


def restore_contract(config, artifact):
    if artifact.get("inference_only"):
        raise ValueError("calibrated inference artifacts cannot resume representation training")
    previous = artifact.get("config", {}).get("experiment", {})
    for key in ("phase", "branch", "config_hash", "manifest_hashes", "structure_hash"):
        if previous.get(key) != config["experiment"][key]:
            raise ValueError(f"resume contract changed: {key}")
    if not artifact.get("rng_streams") or artifact.get("scheduler") is None:
        raise ValueError("large-scale resume needs all random streams and the scheduler")


def run_phase(phase, branch, manifest, splits, resume=False):
    config = phase_config(phase, branch, manifest, splits)
    output, checkpoint = prepare_phase(config, splits, resume)
    snapshot = gpu_snapshot()
    write_json(output / "gpu_at_start.json", snapshot)
    seed_everything(config["seed"], config["device"], config["gpu_memory_fraction"])
    model, _, artifact = build_model(config, config["device"],
                                     checkpoint_override=str(checkpoint) if resume and checkpoint.exists() else None)
    if resume and checkpoint.exists():
        restore_contract(config, artifact)
    actual_resume = resume and checkpoint.exists()
    reconcile_training_logs(config, artifact["step"] if actual_resume else 0)
    return execute_phase(model, config, manifest, splits, artifact, actual_resume)


def prepare_phase(config, splits, resume):
    phase = config["experiment"]["phase"]
    train, settings = config["training"], config["large_scale"]
    if phase != "pilot":
        blockers = scale_conditions(splits, settings, train["crop"])
        if blockers:
            raise MissingPrerequisites(blockers + ["only a labelled finite pilot is allowed at this dataset scale"])
    snapshot = gpu_snapshot()
    fraction = memory_fraction(snapshot, settings)
    output = resolve(config["paths"]["output"])
    previous = output / "config.json"
    reuse_phase_configuration(config, previous, fraction)
    checkpoint = resolve(config["paths"]["save_checkpoint"])
    if checkpoint.exists() and not resume:
        raise ValueError("phase already has a checkpoint; use --resume or a new run-id")
    preflight(config, "train")
    write_json(previous, config)
    return output, checkpoint


def reuse_phase_configuration(config, previous, fraction):
    if previous.is_file():
        stored = read_json(previous, {})
        config["gpu_memory_fraction"] = stored["gpu_memory_fraction"]
        if fraction < stored["gpu_memory_fraction"]:
            raise MissingPrerequisites(["GPU free-memory headroom fell below the recorded training limit"])
        # Keep the original identity for checkpoints made before canonical budget-key hashing.
        config["experiment"]["config_hash"] = stored["experiment"]["config_hash"]
        if stable_hash(stored) != stable_hash(config):
            raise ValueError("refusing to overwrite a different phase configuration; choose a new run-id")
    else:
        config["gpu_memory_fraction"] = fraction
        config["experiment"]["config_hash"] = stable_hash(config)


def execute_phase(model, config, manifest, splits, artifact, actual_resume):
    output = resolve(config["paths"]["output"])
    checkpoint = resolve(config["paths"]["save_checkpoint"])
    observer = TrainingObserver(config, splits, manifest)
    try:
        report = run_training(model, config, config["experiment"]["target_updates"], artifact, actual_resume,
                              status="trained", observer=observer)
        # State saved at the completed optimizer boundary is authoritative.
        from cp_adaptok.runtime import load_checkpoint

        final = load_checkpoint(checkpoint)
        state = final.get("training_state") or {}
        if final_validation_needed(state, final["step"], config["training"]):
            observer.validate(model, final["step"], checkpoint, state, final=True)
            final["training_state"] = state
            temporary = checkpoint.with_suffix(".validation.tmp")
            torch.save(final, temporary)
            temporary.replace(checkpoint)
            observer.checkpoint(final["step"], checkpoint, state)
        observer._phase_state(report["completed_steps"], state, "complete", checkpoint)
        identity = config["experiment"]
        report.update(phase=identity["phase"], branch=identity["branch"], pilot=identity["phase"] == "pilot",
                      convergence_verified=False)
        if report["updates_this_session"] or not (output / "phase_report.json").exists():
            write_json(output / "phase_report.json", report)
        return report
    except BaseException as error:
        write_json(output / "interruption.json", {"error": type(error).__name__, "reason": str(error),
                                                  "checkpoint": str(checkpoint),
                                                  "planned_updates": config["experiment"]["target_updates"],
                                                  "completed": False})
        raise
    finally:
        del model
        torch.cuda.empty_cache()


def final_validation_needed(state, step, training):
    expected = {"quick"}
    if step % training["full_validate_every"] == 0:
        expected.add("full")
    completed = {row["tier"] for row in state.get("validations", []) if row["step"] == step}
    return not expected.issubset(completed)


def run_all(manifest, splits, resume=False):
    for phase in PHASES:
        for branch in (("shared",) if phase == "shared_warmup" else ("B0", "B1")):
            state = manifest["phases"].get(f"{phase}/{branch}", {})
            if state.get("status") == "complete":
                continue
            run_phase(phase, branch, manifest, splits, resume=resume)
    return manifest
