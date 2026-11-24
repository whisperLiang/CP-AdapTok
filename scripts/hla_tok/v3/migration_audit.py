"""Select immutable B0 checkpoints and replay a gradient observation without an update."""

import json
from itertools import islice
from types import SimpleNamespace

from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import FullFrameQuality, sha256_file
from hla_tok.training.checkpoint import read_checkpoint
from hla_tok.training.losses import ReconstructionLoss
from hla_tok.training.train import TrainingFrames
from hla_tok.training.v2 import state_digest
from scripts.hla_tok.v2.protocol import csv_at, frozen_data, json_at, setup
from scripts.hla_tok.v3.calibration import _audit_sample
from scripts.hla_tok.v3.diagnostics import _write_pixel_reads, diagnose
from scripts.hla_tok.v3.downstream import _approved_checkpoint, model_from_checkpoint
from scripts.hla_tok.v3.experiments import on_gpu, require_pass, state_of
from scripts.hla_tok.v3.protocol import CHECKPOINTS, OUTPUT, Trial, curriculum, extend_chain
from scripts.hla_tok.v3.training import _identity, _teacher_weight, configure

MODULES = ("encoder", "query_encoder", "local_decoder", "rgb_decoder")


def _checkpoint_path(trial, folder, step):
    latest = json_at(folder / "latest.json")
    if step is None or step == latest["step"]:
        return _approved_checkpoint({"selected": latest})
    if not 1 <= step <= latest["step"]:
        raise ValueError("the requested B0 checkpoint has not been published")
    path = CHECKPOINTS / trial.relative / f"step_{step:05d}.pt"
    if not path.is_file():
        raise ValueError("the requested update has no saved B0 checkpoint")
    return path


def _optimizer_modules(model, state):
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    group_ids = [index for group in state["optimizer"]["param_groups"] for index in group["params"]]
    if len(trainable) != len(group_ids) or set(group_ids) != set(state["optimizer"]["state"]):
        raise ValueError("saved AdamW state differs from the registered trainable modules")
    return {module: sum(name.startswith(module + ".") for name in trainable) for module in MODULES}


def _digest(module):
    state = module.state_dict()
    return state_digest(state, state)


def _gradient_sample(trial, state):
    path = OUTPUT / "plans" / f"continuous_seed{trial.seed}.jsonl"
    chain, sample = bytes(32).hex(), None
    with path.open() as plan:
        for line in islice(plan, state["step"]):
            sample = json.loads(line)
            chain = extend_chain(chain, sample)
    if sample is None or chain != state["counters"]["sample_chain"]:
        raise ValueError("gradient observation disagrees with the checkpoint's Train sample chain")
    return sample


def _gradient_observation(model, teacher, reader, state, config, trial, weights):
    leg = curriculum(config, trial.stage)[state["counters"]["leg"]]
    configure(model, trial, leg)
    teacher.eval().requires_grad_(False)
    model.train()
    step = state["step"]
    sample = _gradient_sample(trial, state)
    if sample["split"] != "train" or sample["crop_xyhw"][2:] != [leg.geometry.width, leg.geometry.height]:
        raise ValueError("gradient observation requires the registered Train crop")
    raw, geometry, _, _ = TrainingFrames(reader, [leg.geometry], trial.seed).load(sample)
    budget = {geometry.scale_budget(sample["native_budget"]): 1.0}
    loss = ReconstructionLoss("cuda", **weights)
    latent_weight = _teacher_weight(config, step, teacher)
    hashes = {part: _digest(getattr(model, part)) for part in MODULES}
    teacher_hash = _digest(teacher)
    result = _audit_sample(model, transfer(raw[None]), loss, sample, budget, teacher, latent_weight)
    if teacher_hash != _digest(teacher) or any(parameter.grad is not None for parameter in teacher.parameters()):
        raise RuntimeError("gradient observation altered the frozen Teacher")
    if hashes != {part: _digest(getattr(model, part)) for part in MODULES}:
        raise RuntimeError("gradient observation changed Student parameters")
    return result | {"step": step, "latent_weight": latent_weight, "freeze_spatial": leg.freeze_spatial,
                     "optimizer_parameter_counts_by_module": _optimizer_modules(model, state),
                     "unchanged_student_module_sha256": hashes, "unchanged_teacher_sha256": teacher_hash,
                     "teacher_has_gradients": False, "teacher_in_optimizer": False}


def _bound_b0(trial, folder, path, dense_hash, step):
    initialization = json_at(folder / "initialization.json")
    if initialization["dense_sha256"] != dense_hash:
        raise ValueError("B0 does not descend from the selected Phase A checkpoint")
    provenance = {key: initialization[key] for key in ("shared_sha256", "dense_sha256", "r2_sha256")}
    loss = SimpleNamespace(weights=initialization["loss_weights"])
    checkpoint = read_checkpoint(path, _identity(trial, loss, provenance))
    if step is not None and checkpoint.state["step"] != step:
        raise ValueError("B0 checkpoint update does not match its published filename")
    return loss.weights


def _scope(folder, step, full):
    output = folder / "bottleneck"
    if step is None:
        return output, False
    subset = not full
    return output / f"step_{step:05d}" / ("trend" if subset else "full"), subset


def run_diagnostics(config, seed, *, step=None, full=False):
    if step is None:
        decision = json_at(state_of("continuous", seed))
        if decision["candidate_quality"] != "FAIL":
            raise ValueError("B0 failure is required for the default endpoint diagnostic; specify --step for a saved update")
    selection = require_pass(state_of("dense", seed))
    dense_path = _approved_checkpoint(selection)
    selected = selection["selected"]
    trial = Trial("continuous", "B0", seed, selected["variant"])
    folder = OUTPUT / trial.relative
    path = _checkpoint_path(trial, folder, step)
    weights = _bound_b0(trial, folder, path, selected["sha256"], step)
    output, subset = _scope(folder, step, full)
    marker = output / "measurement.json"
    identity = {"b0_path": str(path), "b0_sha256": sha256_file(path),
                "dense_path": str(dense_path), "dense_sha256": selected["sha256"],
                "native_budget": config["native_budget"], "device": "cuda", "precision": "bf16",
                "validation_frames": 9 if subset else 174}
    if marker.is_file():
        if json_at(marker) != identity:
            raise ValueError("an existing diagnostic belongs to different checkpoints")
        return csv_at(output / "metrics.csv")

    setup(seed)
    dense, _ = model_from_checkpoint(dense_path)
    model, state = model_from_checkpoint(path)
    dense, model = on_gpu(dense), on_gpu(model)
    reader, cohort, frozen = frozen_data()
    quality = FullFrameQuality("cuda", frozen["metric_weights"])
    rows = diagnose(dense, model, reader, cohort, quality, config["native_budget"], output, subset=subset)
    before = dict(reader.counts)
    gradient = output / "gradient_audit"
    try:
        result = _gradient_observation(model, dense.encoder, reader, state, config, trial, weights)
        write_json(gradient / "measurement.json", result)
    finally:
        _write_pixel_reads(gradient, {key: reader.counts[key] - before[key] for key in before})
    write_json(marker, identity)
    return rows
