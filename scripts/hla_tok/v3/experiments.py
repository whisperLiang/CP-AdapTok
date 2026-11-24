"""Gated matched experiment orchestration and strict checkpoint migration."""

from pathlib import Path

import torch

from hla_tok import HLATok
from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import FullFrameQuality, sha256_file
from hla_tok.models.spatial import DenseReconstruction, RGBDecoder, SpatialEncoder
from hla_tok.training.losses import ReconstructionLoss
from hla_tok.training.v2 import load_experimental_checkpoint, state_digest
from scripts.hla_tok.v2.benchmark import pipeline
from scripts.hla_tok.v2.protocol import COHORT, frozen_data, json_at, setup
from scripts.hla_tok.v3.benchmark import dense_benchmark, native_references
from scripts.hla_tok.v3.calibration import calibration
from scripts.hla_tok.v3.gates import quality_pass, select_dense, supports_extension
from scripts.hla_tok.v3.inputs import r2_source
from scripts.hla_tok.v3.protocol import OUTPUT, PROTOCOL, Trial, curriculum
from scripts.hla_tok.v3.training import TrialRun, execute


def require_pass(path: Path):
    if not path.is_file() or (value := json_at(path))["status"] != "PASS":
        raise RuntimeError(f"not admitted: {path}")
    return value


def legacy_model(source):
    model = HLATok("R2")
    load_experimental_checkpoint(source, model, sha256_file(COHORT))
    return model


def _expand_decoder(source):
    original = source.state_dict()
    decoder = RGBDecoder("D1")
    combined = decoder.state_dict()
    if not original.keys() < combined.keys() or {key.rsplit(".", 2)[0] for key in combined.keys() - original.keys()} != {
        "upsample.2.2", "upsample.5.2"
    }:
        raise ValueError("D1 does not extend the exact legacy RGB decoder keys")
    combined.update(original)
    decoder.load_state_dict(combined, strict=True)
    return decoder


def dense_model(source, variant):
    original = legacy_model(source)
    rgb_decoder = original.rgb_decoder if variant == "D0" else _expand_decoder(original.rgb_decoder)
    reference = DenseReconstruction(original.encoder, original.rgb_decoder).state_dict()
    model = DenseReconstruction(original.encoder, rgb_decoder)
    return model, state_digest(reference, reference.keys())


def on_gpu(model):
    return model.cuda().to(memory_format=torch.channels_last)


def state_of(phase, seed):
    return OUTPUT / "decisions" / f"{phase}_seed{seed}.json"


def _loss(config, reader, calibration_source):
    path = OUTPUT / "gradient_calibration" / "dense.json"
    if path.is_file():
        audit = json_at(path)
        if audit["protocol_sha256"] != sha256_file(PROTOCOL):
            raise RuntimeError("calibration does not match the frozen protocol")
        return ReconstructionLoss("cuda", **audit["weights"])
    loss = ReconstructionLoss("cuda", **config["loss"])
    before = dict(reader.counts)
    model, _ = dense_model(calibration_source, "D0")
    calibration(on_gpu(model), reader, loss, config, path.with_name("dense_measurements.json"))
    audit = json_at(path.with_name("dense_measurements.json"))
    write_json(path.parent / "pixel_reads.json", {key: reader.counts[key] - before[key] for key in before})
    write_json(path, {"protocol_sha256": sha256_file(PROTOCOL), "weights": audit["weights"],
                      "calibration_initializer_sha256": sha256_file(calibration_source)})
    return loss


def _matching(runs):
    records = [json_at(run.output / name) for name in ("training.json", "initialization.json") for run in runs]
    count = len(runs)
    for key in ("sample_chain", "seen_pixels", "optimizer_updates", "budget_updates"):
        if any(row[key] != records[0][key] for row in records[:count]):
            raise RuntimeError(f"matched trials diverged on {key}")
    if any(row["shared_sha256"] != records[count]["shared_sha256"] for row in records[count:]):
        raise RuntimeError("matched trials did not share their frozen initialization")


def _extension(runs, config, thresholds):
    root = extension_path(runs[0].trial.stage, runs[0].trial.seed)
    if root.is_file():
        return json_at(root)["authorized"]
    checks = {}
    for run in runs:
        paths = sorted((run.output / "full").glob("step_*/metrics.json"))
        history = [json_at(path)[0] for path in paths]
        checks[run.trial.arm] = supports_extension(history, thresholds, config["extension"]["near"])
    authorized = any(checks.values())
    write_json(root, {"authorized": authorized, "eligible_arms": checks,
                      "updates_per_arm": config["extension"]["max_updates"] if authorized else 0})
    return authorized


def extension_path(stage, seed):
    return OUTPUT / "decisions" / f"{stage}_seed{seed}_extension.json"


def extension_active(stage, seed):
    path = extension_path(stage, seed)
    return path.is_file() and json_at(path)["authorized"]


def _selected(run):
    return json_at(run.output / "latest.json") | {"arm": run.trial.arm, "variant": run.trial.variant}


def _d1_working_budget(config, dense, source, reader, cohort):
    model = HLATok("R2", rgb_variant="D1")
    anchor = legacy_model(source)
    model.query_encoder.load_state_dict(anchor.query_encoder.state_dict(), strict=True)
    model.quantizer.load_state_dict(anchor.quantizer.state_dict(), strict=True)
    model.local_decoder.load_state_dict(anchor.local_decoder.state_dict(), strict=True)
    model.encoder.load_state_dict(dense.encoder.state_dict(), strict=True)
    model.rgb_decoder.load_state_dict(dense.rgb_decoder.state_dict(), strict=True)
    controls = config["performance"]
    references = native_references(cohort, controls["frames"])
    before = dict(reader.counts)
    on_gpu(model).eval()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        results = pipeline(model, reader, references, config["native_budget"], controls["repeats"])
    write_json(OUTPUT / "dense" / "d1_working_budget" / "pixel_reads.json",
               {key: reader.counts[key] - before[key] for key in before})
    return results | {"scope": "performance only, legacy uncalibrated codebook"}


def _dense_initializer(v2, seed, r2_override, variants):
    preflight = require_pass(OUTPUT / "preflight.json")
    replay = require_pass(OUTPUT / "baseline" / "replay.json")
    source, _ = r2_source(v2)
    if replay["source_sha256"] != sha256_file(source) or preflight["protocol_sha256"] != sha256_file(PROTOCOL):
        raise RuntimeError("replay or preflight no longer matches the frozen V3 input")
    if seed == 2 and (r2_override is None or len(variants) != 1 or sha256_file(r2_override) == sha256_file(source)):
        raise ValueError("Seed 2 must have independent R2 weights and reuse the selected architecture")
    return source, r2_override or source


def _dense_runs(config, seed, variants, initializer, loss, reader, cohort, quality):
    init_hash = sha256_file(initializer)
    for variant in variants:
        setup(seed)
        model, shared = dense_model(initializer, variant)
        trial = Trial("dense", "A0" if variant == "D0" else "A1", seed, variant)
        yield TrialRun(trial, config, model, loss, None, {"shared_sha256": shared, "r2_sha256": init_hash},
                       reader, cohort, quality)


def _measure_dense_trials(runs, extending):
    results = {}
    for run in runs:
        on_gpu(run.model)
        rows = execute(run, extend=extending)
        run.model.cpu()
        if rows is None:
            return None
        results[run.trial.arm] = rows[0]
    _matching(runs)
    return results


def _dense_parameters(runs):
    parameters = {}
    for run in runs:
        arm = run.trial.arm
        target = run.output / "performance"
        if not (target / "timings.csv").exists():
            dense_benchmark(on_gpu(run.model), run.reader, run.cohort, run.config, target)
            run.model.cpu()
        parameters[arm] = sum(param.numel() for param in run.model.parameters())
    return parameters


def dense_trials(config, v2, seed=1, *, r2_override=None, variants=("D0", "D1")):
    setup(seed)
    source, initializer = _dense_initializer(v2, seed, r2_override, variants)
    reader, cohort, identity = frozen_data()
    quality = FullFrameQuality("cuda", identity["metric_weights"])
    loss = _loss(config, reader, source)
    runs = list(_dense_runs(config, seed, variants, initializer, loss, reader, cohort, quality))
    extending = extension_active("dense", seed)
    results = _measure_dense_trials(runs, extending)
    if results is None:
        return None
    thresholds = config["dense_thresholds"]
    if not extending and not any(quality_pass(row, thresholds) for row in results.values()):
        if _extension(runs, config, thresholds):
            results = _measure_dense_trials(runs, True)
            if results is None:
                return None
    by_arm = {run.trial.arm: run for run in runs}
    selected, perf = select_dense(
        results, config, lambda: _d1_working_budget(config, by_arm["A1"].model, initializer, reader, cohort))
    decision = {"status": "PASS" if selected else "FAIL", "thresholds": thresholds, "observed": results,
                "selected": _selected(by_arm[selected]) if selected else None, "parameters": _dense_parameters(runs)}
    if perf is not None:
        decision["d1_working_budget"] = perf
    write_json(state_of("dense", seed), decision)
    return decision


def empty_dense(variant):
    return DenseReconstruction(SpatialEncoder(), RGBDecoder(variant))


def terminal_step(config, stage):
    return sum(leg.updates for leg in curriculum(config, stage))
