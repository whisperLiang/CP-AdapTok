"""Single-writer training statistics, validation selection, and checkpoint registry."""

import csv
import json
import os
import shutil
from pathlib import Path

import numpy as np
import torch

from cp_adaptok.large_scale_data import file_hash
from cp_adaptok.large_scale_validation import selection_score, validate_checkpoint
from cp_adaptok.runtime import resolve, write_json
from cp_adaptok.training_state import TrainingRNGStreams

TRAIN_FIELDS = ["run_id", "phase", "branch", "step", "lr", "loss", "mse", "i_mse", "p_mse", "grad_norm",
                "vq_perplexity", "vq_usage", "frames", "decoder_calls", "wall_s", "gpu_timeline_s",
                "frames_per_s", "peak_allocated_mib", "peak_reserved_mib"]
VALID_FIELDS = ["run_id", "phase", "branch", "step", "tier", "tokens", "mse", "psnr", "lpips", "dists",
                "ms-ssim", "bpp", "i_psnr", "p_psnr", "deterministic_perplexity", "quality_drop_ratio", "score"]


def append_csv(path, fields, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def read_json(path, default):
    return json.loads(path.read_text()) if path.is_file() else default


def snapshot(source, target):
    """A later atomic replacement of latest cannot modify this immutable inode."""
    target = Path(target)
    temporary = target.with_suffix(".tmp")
    temporary.unlink(missing_ok=True)
    try:
        os.link(source, temporary)
    except OSError:
        shutil.copy2(source, temporary)
    temporary.replace(target)
    # POSIX rename can be a no-op when both names already refer to the same inode.
    temporary.unlink(missing_ok=True)


def mean(values):
    return float(np.mean(values)) if values else None


def budget_values(records, name):
    return [label[name] for row in records for label in row["budgets"] if name in label]


def frame_mse(records, kind):
    return mean(budget_values([r for r in records if r["frame_type"] == kind], "mse"))


def training_statistics(records):
    return {"loss": mean(budget_values(records, "total_loss")), "mse": mean(budget_values(records, "mse")),
            "i_mse": frame_mse(records, "I"), "p_mse": frame_mse(records, "P"),
            "vq_perplexity": mean([r["vq_perplexity"] for r in records]),
            "vq_usage": mean([r["vq_usage"] for r in records])}


def reconcile_training_logs(config, completed):
    """Keep evidence from uncheckpointed updates, without double-counting after resume."""
    output = resolve(config["paths"]["output"])
    identity = config["experiment"]
    for name in ("train.jsonl", "samples.jsonl"):
        path = output / name
        if path.exists():
            retained, aborted = [], []
            for line in path.read_text().splitlines(keepends=True):
                try:
                    keep = json.loads(line)["step"] <= completed
                except (ValueError, KeyError):
                    keep = False
                (retained if keep else aborted).append(line)
            preserve_aborted(path, retained, aborted)
    paths = [output / "training_metrics.csv", resolve(config["large_scale"]["output_root"]) / "training_metrics.csv",
             output / "validation_metrics.csv", resolve(config["large_scale"]["output_root"]) / "validation_metrics.csv"]
    for path in paths:
        reconcile_csv(path, identity, completed)


def reconcile_csv(path, identity, completed):
    if not path.exists():
        return
    lines = path.read_text().splitlines(keepends=True)
    rows = list(csv.DictReader(lines))
    retained, aborted = lines[:1], []
    for line, row in zip(lines[1:], rows):
        same = all(row[key] == identity[key] for key in ("run_id", "phase", "branch"))
        (aborted if same and int(row["step"]) > completed else retained).append(line)
    preserve_aborted(path, retained, aborted)


def preserve_aborted(path, retained, aborted):
    if not aborted:
        return
    backup = path.with_name(path.name + ".uncheckpointed.jsonl")
    with backup.open("a") as handle:
        handle.writelines(aborted)
    temporary = path.with_suffix(path.suffix + ".reconcile")
    temporary.write_text("".join(retained))
    temporary.replace(path)


class TrainingObserver:
    def __init__(self, config, splits, manifest):
        self.config, self.splits, self.manifest = config, splits, manifest
        self.identity = {k: config["experiment"][k] for k in ("run_id", "phase", "branch")}
        self.output = resolve(config["paths"]["output"])
        self.root = resolve(config["large_scale"]["output_root"])
        self.checkpoints = resolve(config["paths"]["save_checkpoint"]).parent

    def update(self, step, records, budgets, norm, lr, elapsed, state):
        frames = len(records)
        decoder_calls = frames * (len(budgets) + 1)
        state["optimizer_updates"] = step
        gpu_s = state.pop("last_gpu_timeline_s", 0)
        for key, value in (("frames", frames), ("decoder_calls", decoder_calls), ("training_wall_s", elapsed),
                           ("gpu_timeline_s", gpu_s)):
            state[key] = state.get(key, 0) + value
        counts = state.setdefault("budget_supervision_frames", {})
        for budget in budgets:
            counts[str(budget)] = counts.get(str(budget), 0) + frames
        self._samples(step, records, state)
        row = self._training_row(step, records, norm, lr, elapsed, decoder_calls)
        row["gpu_timeline_s"] = gpu_s
        append_csv(self.root / "training_metrics.csv", TRAIN_FIELDS, row)
        append_csv(self.output / "training_metrics.csv", TRAIN_FIELDS, row)
        self._phase_state(step, state, "running")

    def _samples(self, step, records, state):
        videos = set(state.get("distinct_videos", []))
        windows = set(state.get("distinct_windows", []))
        with (self.output / "samples.jsonl").open("a") as handle:
            for row in records:
                if row["frame"] != 0:
                    continue
                identity = row.get("content_id") or row["source"]
                window = f"{identity}:{row['window_start']}:{row['window_length']}"
                videos.add(identity)
                windows.add(window)
                sample = {**self.identity, "step": step, "content_id": identity, "path": row["source"],
                          "start": row["window_start"], "frames": row["window_length"],
                          "augmentation": row["augmentation"], "shape": row["input_shape"]}
                handle.write(json.dumps(sample) + "\n")
        state.update(distinct_videos=sorted(videos), distinct_windows=sorted(windows))

    def _training_row(self, step, records, norm, lr, elapsed, calls):
        row = {**self.identity, **training_statistics(records), "step": step, "lr": lr,
               "grad_norm": norm, "frames": len(records),
               "decoder_calls": calls, "wall_s": elapsed, "frames_per_s": len(records) / elapsed}
        if torch.cuda.is_available():
            row.update(peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                       peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
        return row

    def checkpoint(self, step, checkpoint, state):
        path = self.root / "checkpoint_registry.json"
        registry = read_json(path, [])
        for old in registry:
            if old["checkpoint"] == str(checkpoint):
                old.update(available=False, superseded_by_step=step)
        row = {**self.identity, "step": step, "checkpoint": str(checkpoint), "sha256": file_hash(checkpoint),
               "available": True,
               "config_hash": self.config["experiment"]["config_hash"],
               "manifest_hashes": self.config["experiment"]["manifest_hashes"]}
        registry = [r for r in registry if not (all(r[k] == self.identity[k] for k in self.identity)
                                               and r["step"] == step)]
        registry.append(row)
        if step in self.config["training"].get("milestones", []):
            milestone = self.checkpoints / f"step_{step:06d}.pt"
            snapshot(checkpoint, milestone)
            row["checkpoint"] = str(milestone)
        write_json(path, registry)
        self._phase_state(step, state, "running", checkpoint)

    def validate(self, model, step, checkpoint, state, final=False):
        train = self.config["training"]
        if model.vq.strategy == "continuous":
            return
        if not final and step % train["validate_every"]:
            return
        tiers = ["quick"]
        if step % train["full_validate_every"] == 0:
            tiers.append("full")
        for tier in tiers:
            output = self.output / f"validation_{step:06d}_{tier}"
            report = validate_checkpoint(model, self.config, self.splits, tier, output, checkpoint)
            score = selection_score(report)
            self._validation_rows(step, tier, score, report)
            # Compare only the same evaluation population and metric set.
            best = state.setdefault("best_by_tier", {})
            if tier not in best or score < best[tier]["score"]:
                target = self.checkpoints / f"best_{tier}.pt"
                snapshot(checkpoint, target)
                best[tier] = {"step": step, "score": score, "checkpoint": str(target), "report": str(output)}
            state.setdefault("validations", []).append({"step": step, "tier": tier, "score": score})
        # Full validation becomes authoritative after the first full validation point.
        selected = state["best_by_tier"].get("full", state["best_by_tier"]["quick"])
        snapshot(selected["checkpoint"], self.checkpoints / "best_validation.pt")
        write_json(self.output / "selection.json", {"rule": self.config["large_scale"]["selection_rule"],
                                                    "lower_is_better": True, "selected": selected,
                                                    "final_test_used": False})
        self._phase_state(step, state, "running", checkpoint)
        return True

    def _validation_rows(self, step, tier, score, report):
        types = report["budget_summary_by_frame_type"]
        for index, row in enumerate(report["budget_summary"]):
            result = {**self.identity, "step": step, "tier": tier, **row,
                      "i_psnr": types["I"][index]["psnr"], "p_psnr": types["P"][index]["psnr"],
                      "quality_drop_ratio": report["quality_drop_ratio"], "score": score,
                      "deterministic_perplexity": report.get("deterministic_perplexity")}
            append_csv(self.root / "validation_metrics.csv", VALID_FIELDS, result)
            append_csv(self.output / "validation_metrics.csv", VALID_FIELDS, result)

    def _phase_state(self, step, state, status, checkpoint=None):
        summary = {k: v for k, v in state.items() if k not in ("distinct_videos", "distinct_windows")}
        summary.update(distinct_videos=len(state.get("distinct_videos", [])),
                       distinct_windows=len(state.get("distinct_windows", [])))
        key = f"{self.identity['phase']}/{self.identity['branch']}"
        self.manifest["current"] = self.identity
        self.manifest.setdefault("phases", {})[key] = {
            **self.identity, "completed_updates": step, "target_updates": self.config["experiment"]["target_updates"],
            "status": status, "statistics": summary,
            "checkpoint": str(checkpoint or self.config["paths"]["save_checkpoint"]),
            "optimizer": "in checkpoint", "scheduler": "in checkpoint", "rng": list(TrainingRNGStreams.names),
            "config_hash": self.config["experiment"]["config_hash"],
            "structure_hash": self.config["experiment"]["structure_hash"],
            "manifest_hashes": self.config["experiment"]["manifest_hashes"],
            "upstream_backbones": self.manifest["upstreams"],
        }
        write_json(self.output / "state.json", self.manifest["phases"][key])
        write_json(self.root / self.identity["run_id"] / "experiment_manifest.json", self.manifest)
        write_json(self.root / "experiment_manifest.json", self.manifest)
