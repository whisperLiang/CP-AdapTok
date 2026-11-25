"""Gated native adapter training and actual packet checkpoint validation."""

import hashlib
import subprocess
from dataclasses import dataclass
from pathlib import Path
from random import Random
from statistics import mean
from time import perf_counter

import numpy as np
import torch
from torch import Tensor

from hla_tok.compression.profiles import PROFILES_BY_LABEL
from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.streaming.compact_wire import deserialize, max_budget
from hla_tok.streaming.continuous import MAX_K, ContinuousReceiver, encode_continuous
from hla_tok.training.checkpoint import ResumeIdentity, read_checkpoint, save_checkpoint
from hla_tok.training.losses import ReconstructionLoss
from scripts.hla_tok.compact.adapter import assert_frozen, budget_choices, crops_for_step, loss_for_budget
from scripts.hla_tok.compact.admission import admit
from scripts.hla_tok.compact.assets import load_asset, save_asset
from scripts.hla_tok.compact.decision import signed_quality, stage_targets
from scripts.hla_tok.compact.inputs import CONFIG, OUTPUT, TRAIN_PLAN, compact_inputs, compact_orders
from scripts.hla_tok.compact.matrix import Case, evaluate, summarize_matrix
from scripts.hla_tok.continuous.inputs import Inputs, locked_output, run_arguments
from scripts.hla_tok.continuous.matrix import compute, groups, representative, validation_frames
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.v2.protocol import ROOT, csv_at, json_at
from scripts.hla_tok.v3.protocol import PROTOCOL as V3_CONFIG
from scripts.hla_tok.v3.protocol import read_config


@dataclass(frozen=True)
class TrainingData:
    native: Tensor
    images: list[Tensor]
    rms_by_rank: Tensor
    probes: list[tuple[str, int, Tensor, Tensor]]


def _data(inputs: Inputs, config: dict, output: Path) -> TrainingData:
    from scripts.hla_tok.compact.cache import load_train_tokens

    cache = load_train_tokens(inputs, config, output)
    fit = json_at(TRAIN_PLAN)["fit"]
    images = [transfer(inputs.reader.read("train", row["video"], row["frame"])[None]) for row in fit]
    with np.load(output / "rank_covariance.npz", allow_pickle=False) as stats:
        variance = np.diagonal(stats["rank_covariance"], axis1=1, axis2=2).mean(axis=1)
    rms = torch.tensor(variance, dtype=torch.float32, device="cuda").sqrt().clamp_min(.001)
    probe = []
    for video, i, source in validation_frames(inputs, representative(inputs.cohort)):
        with compute(config["compute_dtype"]):
            native = encode_continuous(inputs.model, source)
        probe.append((video, i, source, native))
    return TrainingData(cache.native()[:cache.fit_frames], images, rms, probe)


def _identity(asset: dict, stage: str, seed: int, output: Path, inputs: Inputs) -> ResumeIdentity:
    paths = sorted((ROOT / "hla_tok/compression").glob("*.py"))
    paths += [ROOT / "hla_tok/streaming" / f"{module}.py" for module in ("compact", "compact_wire", "continuous")]
    paths += [Path(__file__).with_name(f"{module}.py") for module in ("adapter", "admission", "train", "assets", "cache")]
    version = hashlib.sha256("".join(sha256_file(path) for path in paths).encode()).hexdigest()
    env = json_at(output / "cache/identity.json")
    bound = hashlib.sha256((sha256_file(CONFIG) + asset["sha256"] + inputs.checkpoint["sha256"]
                            + sha256_file(V3_CONFIG) + str(torch.__version__) + env["compute_dtype"]).encode()).hexdigest()
    data = hashlib.sha256((sha256_file(TRAIN_PLAN) + sha256_file(output / "cache/identity.json")).encode()).hexdigest()
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    return ResumeIdentity(f"{stage}/seed{seed}", bound, data, commit, version)


class Training:
    def __init__(self, inputs: Inputs, config: dict, output: Path, data: TrainingData, spec: dict,
                 name: str, stage: str, seed: int):
        self.inputs = inputs
        self.config = config
        self.output = output
        self.data = data
        self.spec = spec
        self.stage = stage
        self.seed = seed
        self.folder = output / "training" / f"{stage}_{name}_seed{seed}"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.name = self.folder.name
        self.training = config["training"]
        self.identity = _identity(spec, stage, seed, output, inputs)
        order_name = config["quantization"]["comparison_primary_order"]
        self.order = compact_orders()[order_name]
        # Inference-mode index tensors must stay out of the differentiable Rank plan.
        self.validation_order = compact_orders()[order_name]
        profile = PROFILES_BY_LABEL[spec["profile"]]
        self.codec, _ = load_asset(output / "profiles" / spec["filename"], spec, profile, self.order,
                                  inputs.checkpoint["state_dict_sha256"])
        self.low = budget_choices(profile, config["selection"]["low_rate_caps"])
        self.budgets = budget_choices(profile, config["frame_wire_byte_caps"])
        self.rms = data.rms_by_rank.index_select(0, self.codec.ranks).reshape(1, -1, 1)
        self.study_order = Random(seed).sample(range(len(data.images)), len(data.images))
        self.budget_rng = torch.Generator().manual_seed(seed + 17)
        torch.manual_seed(seed)
        if self.codec.transform.lift is None or self.codec.transform.projection is None:
            raise ValueError("adapter requires a pre-trained PCA basis")
        self.codec.transform.projection.requires_grad_(stage == "T1")
        self.codec.transform.lift.requires_grad_(True)
        self.parameters = [p for p in self.codec.transform.parameters() if p.requires_grad]
        legacy = read_config(V3_CONFIG)
        self.loss = ReconstructionLoss("cuda", **legacy["loss"])
        self.quality = NativeQuality()
        self.optimizer = torch.optim.AdamW(self.parameters, lr=self.training["learning_rate"][stage],
                                           weight_decay=legacy["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda _: 1)
        self.clip = legacy["gradient_clip_norm"]

    def update(self, step: int) -> dict:
        index = self.study_order[(step - 1) % len(self.study_order)]
        budget = self.budgets[int(torch.rand((), generator=self.budget_rng).item() * len(self.budgets))]
        self.optimizer.zero_grad(set_to_none=True)
        start = perf_counter()
        with compute(self.config["compute_dtype"]):
            loss, latent = loss_for_budget(
                self.inputs.model, self.codec, self.loss, self.data.images[index],
                self.data.native[index:index + 1], self.rms, budget, crops_for_step(self.seed, step),
                soft=self.stage == "T1", latent_weight=self.training["latent_weight"])
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(self.parameters, self.clip, error_if_nonfinite=True)
        if not torch.isfinite(loss):
            raise FloatingPointError("nonfinite Compact training loss")
        self.optimizer.step()
        self.scheduler.step()
        torch.cuda.synchronize()
        return {"step": step, "budget": budget, "train_index": index, "train_loss": float(loss.detach()),
                "latent_mse_normalized": float(latent.detach()), "gradient_norm": float(norm),
                "wall_ms": 1000 * (perf_counter() - start)}

    def checked_state(self) -> str:
        if self.inputs.reader.counts["test"]:
            raise RuntimeError("Test pixel access was detected")
        return assert_frozen(self.inputs.model, self.inputs.checkpoint["state_dict_sha256"])

    def publish_asset(self, step: int) -> dict:
        path = self.output / "profiles" / f"{self.name}_step{step}.npz"
        return save_asset(path, self.codec.profile, self.spec["projection"], self.codec.transform, self.codec.scalar)

    @torch.inference_mode()
    def validate(self, step: int, spec: dict) -> list[dict]:
        checked = self.checked_state()
        codec, _ = load_asset(self.output / "profiles" / spec["filename"], spec, self.codec.profile, self.validation_order, checked)
        receiver = ContinuousReceiver.from_model(self.inputs.model)
        budgets = sorted({*self.low, MAX_K})
        result: list[dict] = []
        for video, index, target, native in self.data.probes:
            self.quality.prepare(target)
            results = self._decode_probe(codec, receiver, native, index, budgets)
            result.extend({"stage": self.stage, "seed": self.seed, "asset": self.name, "step": step,
                           "video": video, "frame": index} | row | self.quality.score(rgb) for rgb, row in results)
        return result

    def _decode_probe(self, codec, receiver, native, frame, budgets):
        precision = self.config["compute_dtype"]
        with compute(precision):
            ordered = codec.order.apply(native)
            u = codec.transform.encode(ordered, codec.ranks)
            full = codec.frame(codec.encode_codes(u), frame)
            pairs = []
            for k in budgets:
                packet = full.serialize_range(0, k)
                byte_row = deserialize(packet).ledger()
                direct = codec.frame(codec.encode_codes(u[:, :k]), frame)
                if direct.serialize_range(0, k) != packet:
                    raise ValueError("adapter changed the actual strict prefix")
                actual = codec.decode(receiver, packet)
                quantized = codec.lift(codec.dequantize(codec.prefix_codes(packet).cuda()))
                latent = float((quantized.float() - ordered[:, :k].float()).square().mean())
                bypass = receiver.decode(codec.lift(u[:, :k]), codec.order)
                pairs.extend([(actual, {"budget": k, "path": "hard_wire", "latent_mse": latent} | byte_row),
                              (bypass, {"budget": k, "path": "continuous_bypass", "latent_mse": ""})])
        return pairs


def _profile_quality(data: list[dict], budget: int | None = None) -> tuple[float, ...]:
    groups_by_budget = list(groups([row for row in data if budget is None or int(row["budget"]) == budget], ("budget",)))
    by_budget = [signed_quality(summarize(rows)) for _, rows in groups_by_budget]
    return tuple(mean(values) for values in zip(*by_budget, strict=True))


def _deltas(new: tuple, old: tuple) -> tuple[float, ...]:
    return tuple(a - b for a, b in zip(new, old, strict=True))


def paired_gate(current: list[dict], previous: list[dict], original: list[dict], config: dict) -> dict:
    paths = ("hard_wire", "continuous_bypass")
    hard, old = ([r for r in rows if r["path"] == paths[0] and int(r["budget"]) != MAX_K]
                 for rows in (current, previous))
    gain = _deltas(_profile_quality(hard), _profile_quality(old))
    keeps_full = _retains_ceiling(current, original, config)
    by_source = {}
    for video in dict.fromkeys(row["video"] for row in hard):
        a, b = ([r for r in rows if r["video"] == video] for rows in (hard, old))
        by_source[video] = _deltas(_profile_quality(a), _profile_quality(b))
    stable = sum(_no_worse(values, config) and any(v > 0 for v in values) for values in by_source.values())
    thresholds = tuple(config[f"minimum_material_gain_{name}"] for name in ("psnr", "lpips", "dists"))
    material = _no_worse(gain, config) and (gain[0] >= thresholds[0] or
                                            (gain[1] >= thresholds[1] and gain[2] >= thresholds[2]))
    return {"adopted": bool(keeps_full and material and stable >= config["source_coverage_minimum"]),
            "hard_low_signed_gains": list(gain), "stable_sources": stable, "by_source": by_source,
            "retains_maximum": keeps_full, "material": material}


def _no_worse(values: tuple, config: dict) -> bool:
    tolerance = config["full_max_loss"]
    return all(delta >= -tolerance[key] for key, delta in zip(("global_psnr", "lpips", "dists"), values, strict=True))


def _retains_ceiling(new: list[dict], base: list[dict], config: dict) -> bool:
    for path in ("hard_wire", "continuous_bypass"):
        a, b = (_profile_quality([r for r in rows if r["path"] == path], MAX_K) for rows in (new, base))
        if not _no_worse(_deltas(a, b), config):
            return False
    return True


def _rows(path: Path) -> list[dict]:
    cast = dict.fromkeys(("frame", "budget", "step", "seed"), int) | dict.fromkeys(("mse", "psnr", "lpips", "dists"), float)
    return [{k: cast.get(k, str)(v) for k, v in row.items()} for row in csv_at(path)]


def _stage(run: Training, max_steps: int, original: list[dict] | None = None) -> dict:
    ready = run.folder / "stage.json"
    if ready.is_file():
        result = json_at(ready)
        if json_at(run.folder / "identity.json") != vars(run.identity):
            raise ValueError("completed training stage identity differs")
        return result
    write_json(run.folder / "identity.json", vars(run.identity))
    checkpoints = sorted(run.folder.glob("step_*.pt"))
    previous = 0
    if checkpoints:
        saved = read_checkpoint(checkpoints[-1], run.identity).restore(
            run.codec.transform, run.optimizer, run.scheduler, run.budget_rng)
        previous = saved["step"]
    probes = run.folder / "validation_00000.csv"
    if not probes.is_file():
        write_csv(probes, run.validate(0, run.publish_asset(0)))
    base = _rows(probes)
    original = base if original is None else original
    record = saved["counters"] if previous else {"pending": 0, "best": 0, "successful": False, "elapsed_update_ms": 0}
    checkpoints_at = sorted({max_steps, *(s for s in run.training["validation_steps"] if 0 < s <= max_steps)})
    log = []
    for step in range(previous + 1, max_steps + 1):
        if record["pending"] >= run.training["stop_patience"] or record.get("rolled_back"):
            break
        row = run.update(step)
        record["elapsed_update_ms"] += row["wall_ms"]
        log.append(row)
        if step not in checkpoints_at:
            continue
        spec = run.publish_asset(step)
        scored = run.validate(step, spec)
        prior = _rows(run.folder / f'validation_{record["best"]:05d}.csv')
        gate = paired_gate(scored, prior, original, run.training)
        record = _save_validation(run, step, scored, gate, record, spec, log)
        log = []
    best_step = int(record["best"])
    result = record | {"best_step": best_step, "base_csv": str(probes.relative_to(run.output)),
                       "best_csv": str((run.folder / f"validation_{best_step:05d}.csv").relative_to(run.output))}
    write_json(ready, result)
    return result


def _save_validation(run: Training, step: int, scored: list[dict], gate: dict, record: dict,
                     spec: dict, update_log: list[dict]) -> dict:
    if gate["adopted"]:
        record.update(best=step, successful=True, pending=0, spec=spec)
    else:
        record["pending"] += 1
    record["rolled_back"] = not gate["retains_maximum"]
    record["steps"] = step
    write_csv(run.folder / f"validation_{step:05d}.csv", scored)
    write_csv(run.folder / f"updates_{step:05d}.csv", update_log)
    write_json(run.folder / f"gate_{step:05d}.json", gate)
    save_checkpoint(run.folder / f"step_{step:05d}.pt", run.identity, run.codec.transform,
                    run.optimizer, run.scheduler, run.budget_rng, step=step, data_cursor=step, counters=record)
    print(f"{run.name} step {step}: gains {gate['hard_low_signed_gains']}; accept={gate['adopted']}", flush=True)
    return record


def _t1_choice(results: list[tuple[str, dict]], output: Path) -> tuple[str, dict]:
    quality = {key: _profile_quality([r for r in _rows(output / stage["best_csv"])
                                      if r["path"] == "hard_wire" and int(r["budget"]) != MAX_K])
               for key, stage in results}
    ranks = {key: sum(a > b for other in quality.values() for a, b in zip(other, ours, strict=True))
             for key, ours in quality.items()}
    return min(results, key=lambda item: ranks[item[0]])


def _final_cases(spec: dict, config: dict) -> list[dict]:
    profile = PROFILES_BY_LABEL[spec["profile"]]
    bands = [{"target_mbps": mbps, "cap": cap, "basis": "total_wire", "budget": max_budget(cap, profile)}
             for mbps, cap in zip(config["bandwidth_mbps"], config["frame_wire_byte_caps"], strict=True)]
    k = max_budget(config["equal_payload"]["cap_bytes"], profile, payload_only=True)
    return bands + [{"target_mbps": "", "cap": config["equal_payload"]["cap_bytes"], "basis": "payload_only", "budget": k}]


def _finalize(inputs: Inputs, config: dict, output: Path, spec: dict, original_key: str, selection: dict) -> dict:
    key = Path(spec["filename"]).stem
    conditions = _final_cases(spec, config)
    wanted = {r["budget"] for r in conditions} | set(config["budgets"])
    rows = evaluate(inputs, config, {key: spec}, output, key, [Case(key, selection["order"], k) for k in wanted])
    summary = summarize_matrix(rows)
    lookup = {row["budget"]: row for row in summary}
    bands = [r | lookup[r["budget"]] for r in conditions]
    for row in bands:
        field = "payload_bytes_per_frame" if row["basis"] == "payload_only" else "total_bytes_per_frame"
        if row[field] > row["cap"]:
            raise ValueError("adapted packet exceeds its evaluated cap")
        row["rate_feasibility"] = "SATURATED" if row["budget"] == MAX_K else "FEASIBLE"
    write_csv(output / "training_final_per_frame.csv", rows)
    write_csv(output / "training_final_rd.csv", summary)
    write_csv(output / "training_final_bandwidth.csv", bands)
    equal = bands[-1]
    decision = stage_targets(bands[:-1], equal, config)
    base = [r for r in _rows(output / "scalar_quantization_per_frame.csv")
            if r["asset"] == original_key and r["order"] == selection["order"] and r["budget"] in wanted]
    hard_low = {r["budget"] for r in bands[:-1] if float(r["target_mbps"]) <= 10} - {MAX_K}
    new_low = [r | {"path": "hard_wire"} for r in rows if r["budget"] in hard_low]
    old_low = [r | {"path": "hard_wire"} for r in base if r["budget"] in hard_low]
    gain = _deltas(_profile_quality(new_low), _profile_quality(old_low))
    full_gain = _deltas(_profile_quality(rows, MAX_K), _profile_quality(base, MAX_K))
    index = {float(r["target_mbps"]): r for r in csv_at(output / "bandwidth_constrained_quality.csv")
             if (r["asset"], r["order"]) == (selection["main_asset"], selection["order"])}
    main_deltas = [_deltas(signed_quality(row), signed_quality(index[float(row["target_mbps"])]))
                   for row in bands[:-1] if float(row["target_mbps"]) <= 10]
    against_main = tuple(mean(items) for items in zip(*main_deltas, strict=True))
    adopted = _no_worse(against_main, config["training"]) and _no_worse(full_gain, config["training"]) and _no_worse(gain, config["training"]) and (
        gain[0] >= config["training"]["minimum_material_gain_psnr"] or
        (gain[1] >= config["training"]["minimum_material_gain_lpips"]
         and gain[2] >= config["training"]["minimum_material_gain_dists"]))
    return {"asset": key, "spec": spec, "parent_asset": original_key, "targets": decision, "adopted": adopted,
            "full_cohort_signed_gains_low": list(gain), "full_cohort_signed_gains_max": list(full_gain),
            "signed_gain_at_low_caps_vs_static_main": list(against_main)}


def _training(inputs: Inputs, config: dict, output: Path, assets: dict, selection: dict, admitted: list[str]) -> dict:
    data = _data(inputs, config, output)
    knobs = config["training"]
    seed = knobs["seeds"][0]
    runs = [(key, _stage(Training(inputs, config, output, data, assets[key], key, "T1", seed), knobs["steps"]["T1"]))
            for key in admitted]
    successes = [r for r in runs if r[1]["successful"]]
    if not successes:
        return {"status": "T1_STOP", "adopted": False, "T1": dict(runs), "T2": "NOT_APPLICABLE",
                "T3": "NOT_APPLICABLE", "second_seed": "NOT_APPLICABLE"}
    key, t1 = _t1_choice(successes, output)
    t2 = _stage(Training(inputs, config, output, data, t1["spec"], key, "T2", seed), knobs["steps"]["T2"],
                _rows(output / t1["base_csv"]))
    spec = t2.get("spec", t1["spec"])
    final = _finalize(inputs, config, output, spec, key, selection)
    probe: str | dict = "NOT_APPLICABLE"
    if final["targets"]["equal_payload_pass"] and final["targets"]["all_practical_pass"]:
        runner = Training(inputs, config, output, data, assets[key], key, "T1", knobs["seeds"][1])
        probe = _stage(runner, knobs["replicate_steps"])
    return final | {"status": "T2_FINISHED", "T1": dict(runs), "T2": t2, "second_seed": probe,
                    "T3": "NOT_APPLICABLE",
                    "T3_reason": "no direct attribution of residual loss to Recovery input weights"}


def _curves(output: Path) -> None:
    records = [r for path in sorted((output / "training").glob("*/validation_*.csv")) for r in _rows(path)]
    if records:
        write_csv(output / "training_validation_per_frame.csv", records)
        write_csv(output / "training_curves.csv", [
            tags | summarize(rows) for tags, rows in groups(records, ("stage", "seed", "asset", "step", "path", "budget"))
        ])


def main() -> None:
    parser = run_arguments(__doc__)
    parser.set_defaults(output=OUTPUT)
    args = parser.parse_args()
    with locked_output(args.output):
        inputs, config = compact_inputs(args.v3, args.output)
        assets = json_at(args.output / "compact_profile_manifest.json")
        result = json_at(args.output / "no_training_selection.json")
        admitted = admit(inputs, config, args.output, assets, result)
        if admitted:
            verdict = _training(inputs, config, args.output, assets, result, admitted)
        else:
            verdict = {"status": "NOT_APPLICABLE", "adopted": False,
                       "reason": "no measured projection headroom or the untrained codec met the frozen targets"}
        _curves(args.output)
        inputs.finish(args.output / "training_inputs.json")
        write_json(args.output / "training_selection.json", verdict)


if __name__ == "__main__":
    main()
