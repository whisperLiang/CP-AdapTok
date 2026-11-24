"""Ordered fixed-continuous, fixed-discrete and progressive curricula."""

from argparse import Namespace
from copy import deepcopy
from pathlib import Path

from hla_tok import BUDGETS, HLATok
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import FullFrameQuality, sha256_file
from hla_tok.training.checkpoint import read_checkpoint
from hla_tok.training.losses import Losses
from hla_tok.training.train import calibrate
from hla_tok.training.v2 import state_digest
from scripts.hla_tok.v2.protocol import COHORT, ROOT, csv_at, frozen_data, json_at, setup
from scripts.hla_tok.v2.train import Trial as LegacyTrial
from scripts.hla_tok.v2.train import execute as run_legacy
from scripts.hla_tok.v3.benchmark import codec_benchmark
from scripts.hla_tok.v3.calibration import calibration
from scripts.hla_tok.v3.diagnostics import diagnose
from scripts.hla_tok.v3.experiments import (
    _extension,
    _loss,
    _matching,
    _selected,
    dense_trials,
    empty_dense,
    extension_active,
    legacy_model,
    on_gpu,
    require_pass,
    state_of,
)
from scripts.hla_tok.v3.gates import (
    discrete_thresholds,
    progressive_decision,
    quality_pass,
    supports_extension,
)
from scripts.hla_tok.v3.inputs import r2_source
from scripts.hla_tok.v3.protocol import CHECKPOINTS, CROP, OUTPUT, Trial
from scripts.hla_tok.v3.training import TrialRun, check_model_metadata, execute, latest_path
from scripts.hla_tok.v3.verify import verify_native

BASELINES = ROOT / "results/hla_tok/baselines/metrics.csv"


def model_from_checkpoint(path):
    state = read_checkpoint(path).state
    if state["identity"]["dataset_sha256"] != sha256_file(COHORT):
        raise ValueError("checkpoint is not bound to the frozen native cohort")
    meta = state["counters"]["architecture"]
    variant = meta.get("rgb_decoder", "D0")
    version = meta["architecture_version"]
    if version not in ("hla-tok.dense", "hla-tok.R2"):
        raise ValueError("V3 only admits the registered dense and R2 architectures")
    model = empty_dense(variant) if version == "hla-tok.dense" else HLATok("R2", rgb_variant=variant)
    check_model_metadata(model, state)
    model.load_state_dict(state["model"], strict=True)
    return model, state


def _approved_checkpoint(decision):
    chosen = decision["selected"]
    path = Path(chosen["path"])
    if not path.is_file() or sha256_file(path) != chosen["sha256"]:
        raise ValueError("a selected parent checkpoint is missing or has changed")
    return path


class SeedTwoTrial(LegacyTrial):
    @property
    def output(self):
        return OUTPUT / "r2_seed2" / "v2_protocol"

    @property
    def checkpoints(self):
        return CHECKPOINTS / "r2_seed2" / "v2_protocol"


def seed_two_r2():
    require_pass(state_of("quantized", 1))
    authorization = OUTPUT / "r2_seed2" / "initialization.json"
    if authorization.is_file():
        approved = require_pass(authorization)
        return _approved_checkpoint(approved)
    import yaml

    setup(2)
    cfg = yaml.safe_load((ROOT / "configs/hla_tok/v2/protocol.yaml").read_bytes())
    stage = cfg["pilot"]
    trial = SeedTwoTrial("pilot", "R2", 2, 2 * stage["updates"], stage["every"], CROP, 0, False)
    resume = max(trial.checkpoints.glob("step_*.pt"), default=None)
    reader, _, identity = frozen_data()
    if reader.counts["test"]:
        raise RuntimeError("Test pixels were seen before model freeze")
    quality = FullFrameQuality("cuda", identity["metric_weights"])
    if resume is None or read_checkpoint(resume).state["step"] < trial.updates:
        run_legacy(trial, cfg, Namespace(resume=resume, initialize_from=None, legacy_root=None),
                   quality, Losses("cuda", **cfg["loss"]))
    path = max(trial.checkpoints.glob("step_*.pt"), default=None)
    if path is None or read_checkpoint(path).state["step"] != trial.updates:
        return None
    selected = {"path": str(path), "sha256": sha256_file(path)}
    write_json(authorization, {"status": "PASS", "selected": selected, "uses_seed1_weights": False,
                               "updates": trial.updates, "source": "V2 original R2 pilot and equal continuation"})
    return path


def _seed_r2(v2, seed):
    if seed == 1:
        return r2_source(v2)[0]
    return _approved_checkpoint(require_pass(OUTPUT / "r2_seed2" / "initialization.json"))


def _context(config, v2, seed):
    setup(seed)
    reader, cohort, identity = frozen_data()
    loss = _loss(config, reader, r2_source(v2)[0])
    quality = FullFrameQuality("cuda", identity["metric_weights"])
    return reader, cohort, loss, quality


def _digest(model):
    state = model.state_dict()
    return state_digest(state, state.keys())


def _make_continuous(r2, dense):
    source = legacy_model(r2)
    model = HLATok("R2", rgb_variant=dense.rgb_decoder.variant)
    for name in ("query_encoder", "local_decoder", "quantizer"):
        getattr(model, name).load_state_dict(getattr(source, name).state_dict(), strict=True)
    model.encoder.load_state_dict(dense.encoder.state_dict(), strict=True)
    model.rgb_decoder.load_state_dict(dense.rgb_decoder.state_dict(), strict=True)
    model.quantizer.requires_grad_(False)
    return model


def _continuous_run(config, v2, seed):
    dense = require_pass(state_of("dense", seed))
    parent = _approved_checkpoint(dense)
    r2 = _seed_r2(v2, seed)
    reader, cohort, loss, quality = _context(config, v2, seed)
    dense_model, _ = model_from_checkpoint(parent)
    model = on_gpu(_make_continuous(r2, dense_model))
    teacher = on_gpu(deepcopy(dense_model.encoder)).eval().requires_grad_(False)
    variant = dense["selected"]["variant"]
    run = TrialRun(Trial("continuous", "B0", seed, variant), config, model, loss, teacher,
                   {"shared_sha256": _digest(model), "dense_sha256": sha256_file(parent),
                    "r2_sha256": sha256_file(r2)}, reader, cohort, quality)
    return run, dense_model


def _audit_continuous_initialization(run):
    audit = run.output / "gradient_calibration.json"
    if audit.is_file():
        return
    before = dict(run.reader.counts)
    calibration(run.model, run.reader, run.loss, run.config, audit, teacher=run.teacher, freeze_weights=True)
    write_json(run.output / "gradient_calibration" / "pixel_reads.json",
               {key: run.reader.counts[key] - before[key] for key in before})


def _continuous_endpoint(run, extending, thresholds):
    rows = execute(run, extend=extending)
    if rows is None:
        return None
    if not extending and not quality_pass(rows[0], thresholds) and _extension([run], run.config, thresholds):
        rows = execute(run, extend=True)
        if rows is None:
            return None
    return rows[0]


def continuous_trials(config, v2, seed):
    run, dense_model = _continuous_run(config, v2, seed)
    _audit_continuous_initialization(run)
    extending = extension_active("continuous", seed)
    thresholds = config["fixed_thresholds"]
    observed = _continuous_endpoint(run, extending, thresholds)
    if observed is None:
        return None
    passed = quality_pass(observed, thresholds)
    decision = {"status": "PASS" if passed else "BLOCKED", "candidate_quality": "PASS" if passed else "FAIL",
                "thresholds": thresholds, "observed": observed, "selected": _selected(run) if passed else None,
                "reason": None if passed else "B0 missed the gate; the conditional matched ARD decision is required"}
    write_json(state_of("continuous", seed), decision)
    if not passed:
        target = run.output / "bottleneck"
        diagnose(on_gpu(dense_model), run.model, run.reader, run.cohort, run.quality, config["native_budget"], target)
    return decision


def _calibrate_quantizer(run):
    if latest_path(run) is not None:
        return
    before = dict(run.reader.counts)
    config = run.config
    calibrate(run.model, run.reader, config["calibration_frames"], config["native_budget"],
              run.trial.seed, run.output / "vq_calibration")
    weight = run.model.quantizer.state_dict()
    write_json(run.output / "vq_calibration" / "codebook_sha256.json", {"sha256": state_digest(weight, weight.keys())})
    write_json(run.output / "vq_calibration" / "pixel_reads.json",
               {key: run.reader.counts[key] - before[key] for key in before})


def _performance_and_contract(run, budgets, precisions):
    path = run.output / "performance"
    model = run.model
    verify_native(model, run.reader, run.cohort, run.config["native_budget"], run.output / "verify")
    rows = codec_benchmark(model, run.reader, run.cohort, run.config, path, budgets, precisions=precisions)
    primary, = (row for row in rows if row["budget"] == run.config["native_budget"] and row["precision"] == "bf16")
    return primary


def _quantized_runs(config, v2, seed):
    continuous = require_pass(state_of("continuous", seed))
    parent = _approved_checkpoint(continuous)
    reader, cohort, loss, quality = _context(config, v2, seed)
    variant = continuous["selected"]["variant"]
    for arm in ("C0", "C1"):
        model, _ = model_from_checkpoint(parent)
        run = TrialRun(Trial("quantized", arm, seed, variant), config, on_gpu(model), loss, None,
                       {"shared_sha256": _digest(model), "continuous_sha256": sha256_file(parent)},
                       reader, cohort, quality)
        if arm == "C1":
            _calibrate_quantizer(run)
        yield run


def _run_observations(runs, extending):
    observations = []
    for run in runs:
        rows = execute(run, extend=extending)
        if rows is None:
            return None
        observations.append((run, rows))
    return observations


def _passes_performance(performance, config):
    return performance is not None and performance["pipeline_fps"] >= config["performance"]["pipeline_fps"]


def _fixed_verdict(observations, thresholds, config):
    (_, (control,)), (candidate, (reported, bypass)) = observations
    perf = None
    if quality_pass(reported, thresholds):
        perf = _performance_and_contract(candidate, (config["native_budget"],), ("bf16",))
    passed = _passes_performance(perf, config)
    return {
        "status": "PASS" if passed else "FAIL", "thresholds": thresholds, "observed": reported,
        "selected": _selected(candidate) if passed else None, "performance": perf,
        "immediate_fixed13_minus_bypass": {key: reported[key] - bypass[key] for key in thresholds},
        "continuous_training_trajectory": {key: bypass[key] - control[key] for key in thresholds},
    }


def quantized_trials(config, v2, seed):
    extending = extension_active("quantized", seed)
    observations = _run_observations(_quantized_runs(config, v2, seed), extending)
    if observations is None:
        return None
    runs = [run for run, _ in observations]
    _matching(runs)
    candidate, (reported, _) = observations[-1]
    thresholds = discrete_thresholds(reported, config, csv_at(BASELINES))
    if not extending and not quality_pass(reported, thresholds) and _extension([candidate], config, thresholds):
        observations = _run_observations(runs, True)
        if observations is None:
            return None
        _matching(runs)
    decision = _fixed_verdict(observations, thresholds, config)
    write_json(state_of("quantized", seed), decision)
    return decision


def _nonmonotonic(run, step):
    source = run.output / "full" / f"step_{step:05d}" / "per_frame.csv"
    frames = csv_at(source)
    by_budget = [{(row["video"], row["frame"]): row for row in frames if int(row["budget"]) == budget}
                 for budget in BUDGETS]
    results = []
    for budget, next_budget, left, right in zip(BUDGETS, BUDGETS[1:], by_budget, by_budget[1:]):
        if left.keys() != right.keys() or len(left) != 174:
            raise ValueError("progressive comparison frames differ")
        row = {"lower_budget": budget, "upper_budget": next_budget, "frames": len(left)}
        for key, direction in (("psnr", 1), ("lpips", -1), ("dists", -1)):
            differences = [direction * (float(right[frame][key]) - float(left[frame][key])) for frame in left]
            row[f"{key}_negative_fraction"] = sum(value < 0 for value in differences) / len(differences)
        results.append(row)
    write_csv(run.output / "frame_nonmonotonicity.csv", results)


def _fixed_trends(run):
    fixed = run.config["native_budget"]
    folders = sorted((run.output / "full").glob("step_*/metrics.json"))
    return [next(row for row in json_at(path) if row["budget"] == fixed) for path in folders]


def _progressive_runs(config, v2, parents):
    for seed, parent in parents.items():
        source = _approved_checkpoint(parent)
        reader, cohort, loss, quality = _context(config, v2, seed)
        model, _ = model_from_checkpoint(source)
        trial = Trial("progressive", "D", seed, parent["selected"]["variant"])
        yield TrialRun(trial, config, on_gpu(model), loss, None,
                       {"shared_sha256": _digest(model), "fixed_vq_sha256": sha256_file(source)},
                       reader, cohort, quality)


def _progressive_extension(path, runs, decisions, config):
    if path.is_file():
        return json_at(path)["authorized"]
    attempts = {run.trial.seed: _fixed_trends(run) for run in runs}
    allowed = any(supports_extension(attempts[seed], verdict["fixed_thresholds"], config["extension"]["near"])
                  for seed, verdict in decisions.items())
    write_json(path, {"authorized": allowed, "updates_per_seed": config["extension"]["max_updates"] if allowed else 0})
    return allowed


def _publish_progressive(run, decision, config):
    selected = _selected(run)
    _nonmonotonic(run, selected["step"])
    if decision["status"] == "PASS":
        perf = _performance_and_contract(run, BUDGETS, ("bf16", "fp32"))
        decision["performance"] = perf
        decision["status"] = "PASS" if _passes_performance(perf, config) else "FAIL"
    decision["selected"] = selected if decision["status"] == "PASS" else None
    write_json(state_of("progressive", run.trial.seed), decision)


def _progressive_verdicts(observations, config, baselines):
    return {run.trial.seed: progressive_decision(rows, config, baselines) for run, rows in observations}


def _progressive_observations(config, v2, parents, extension):
    extending = extension.is_file() and json_at(extension)["authorized"]
    observations = _run_observations(_progressive_runs(config, v2, parents), extending)
    if observations is None:
        return None
    runs = [run for run, _ in observations]
    baselines = csv_at(BASELINES)
    decisions = _progressive_verdicts(observations, config, baselines)
    if not _progressive_extension(extension, runs, decisions, config) or extending:
        return runs, decisions
    extra = _run_observations(runs, True)
    return None if extra is None else (runs, _progressive_verdicts(extra, config, baselines))


def progressive_trials(config, v2):
    parents = {seed: require_pass(state_of("quantized", seed)) for seed in (1, 2)}
    extension = OUTPUT / "decisions" / "progressive_extension.json"
    observations = _progressive_observations(config, v2, parents, extension)
    if observations is None:
        return None
    runs, decisions = observations
    for run in runs:
        _publish_progressive(run, decisions[run.trial.seed], config)
    return {"status": "PASS" if all(row["status"] == "PASS" for row in decisions.values()) else "FAIL",
            "seeds": decisions}


def _run_if_pending(phase, seed, work):
    path = state_of(phase, seed)
    return json_at(path) if path.is_file() else work()


def full_pipeline(config, v2):
    for seed in (1, 2):
        initial = None
        selected = None
        if seed == 2:
            initial = seed_two_r2()
            if initial is None:
                return {"status": "PARTIAL PASS"}
            selected = require_pass(state_of("dense", 1))["selected"]["variant"]
        def dense_work():
            if selected is None:
                return dense_trials(config, v2)
            return dense_trials(config, v2, 2, r2_override=initial, variants=(selected,))
        for phase, work in (("dense", dense_work), ("continuous", lambda: continuous_trials(config, v2, seed)),
                            ("quantized", lambda: quantized_trials(config, v2, seed))):
            decision = _run_if_pending(phase, seed, work)
            if decision is None or decision["status"] != "PASS":
                return decision
    return progressive_trials(config, v2)
