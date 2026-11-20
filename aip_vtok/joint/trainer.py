"""One optimizer update per native causal window."""

import math
import random
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from aip_vtok.decoder.causal_state import ReferenceState
from aip_vtok.diagnostics.diag0.closed import same_state
from aip_vtok.training.losses import crop_plan

from .bitstream import GOP
from .codec import JointCodec
from .loss import JointLoss
from .scan import BUDGETS, REF_BUDGET, SLOT_COUNT
from .tokenizer import GROUP_WIDTH

LOW = BUDGETS[:5]
HIGH = BUDGETS[4:]


@dataclass(frozen=True)
class Window:
    video: str
    start: int
    length: int

    @property
    def anchor(self) -> int:
        return self.start // GOP * GOP

    @property
    def span(self) -> int:
        return self.start - self.anchor + self.length


def window_for(rows: dict, seed: int, step: int, length: int) -> Window:
    eligible = sorted(v for v, r in rows.items() if r["original_frames"] >= length)
    generator = np.random.default_rng(np.random.SeedSequence([seed, step]))
    video = eligible[int(generator.integers(len(eligible)))]
    start = int(generator.integers(rows[video]["original_frames"] - length + 1))
    return Window(video, start, length)


class BudgetSampler:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)
        self.pending: dict[str, list[int]] = {f"{kind}_{tier}": [] for kind in ("I", "P") for tier in ("lo", "hi")}

    def draw(self, intra: bool) -> tuple[int, ...]:
        kind = "I" if intra else "P"
        picked = [REF_BUDGET[intra]]
        for tier, budgets in (("lo", LOW), ("hi", HIGH)):
            pending = self.pending[f"{kind}_{tier}"]
            if not pending:
                pending.extend(budgets)
                self.rng.shuffle(pending)
            picked.append(pending.pop())
        return tuple(dict.fromkeys(picked))

    def state_dict(self) -> dict:
        return {"rng": self.rng.getstate(), "pending": {k: v.copy() for k, v in self.pending.items()}}

    def load_state_dict(self, state: dict) -> None:
        self.rng.setstate(state["rng"])
        self.pending = {k: v.copy() for k, v in state["pending"].items()}


def lr_factor(step: int, config: dict) -> float:
    upcoming = min(step + 1, config["steps"])
    warmup = config["warmup_steps"]
    if upcoming <= warmup:
        return upcoming / warmup
    progress = (upcoming - warmup) / (config["steps"] - warmup)
    floor = config["cosine_end_factor"]
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


@dataclass
class History:
    tx: ReferenceState
    rx: ReferenceState

    @classmethod
    def at(cls, anchor: int):
        if anchor < 0 or anchor % GOP:
            raise ValueError("window anchor must start at an actual I frame")
        return cls(ReferenceState(frame_id=anchor - 1), ReferenceState(frame_id=anchor - 1))

    def submit(self, tx: ReferenceState, rx: ReferenceState) -> None:
        if not same_state(tx, rx):
            raise RuntimeError(f"integer reference states differ at frame {tx.frame_id}")
        self.tx, self.rx = tx, rx


class Trainer:
    def __init__(self, codec: JointCodec, config: dict, seed: int, objective: JointLoss | None = None):
        self.codec, self.config, self.seed = codec, config, seed
        self.objective = objective if objective is not None else JointLoss(config["loss"]).to(
            device=codec.device, memory_format=torch.channels_last
        )
        self.budgets = BudgetSampler(seed + 8001)
        groups = [{"params": list(getattr(codec, name).parameters()), "lr": lr, "name": name}
                  for name, lr in config["lr"].items()]
        self.optimizer = torch.optim.AdamW(groups, weight_decay=config["weight_decay"])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lambda step: lr_factor(step, config))
        self.parameters = [p for group in self.optimizer.param_groups for p in group["params"]]

    def stage(self, step: int) -> str:
        if step <= self.config["phase_a_end"]:
            return "A"
        return "B" if step <= self.config["phase_b_end"] else "C"

    def schedule(self, step: int) -> str:
        stage = self.stage(step)
        self.codec.transform.requires_grad_(stage != "A")
        self.codec.entropy.requires_grad_(stage != "A")
        if step >= self.config["scan_freeze_step"] and not self.codec.scan.frozen:
            self.codec.scan.freeze()
        self.codec.scan.requires_grad_(stage != "A" and not self.codec.scan.frozen)
        return stage

    def next_window(self, rows, step):
        if step > self.config["steps"]:
            return None
        length = self.config["window_c"] if self.stage(step) == "C" else self.config["window_ab"]
        return window_for(rows, self.seed, step, length)

    @torch.no_grad()
    def encode(self, rgb, history, frame_id):
        backbone = self.codec.backbone
        tx = backbone.begin_frame(history.tx, frame_id)
        rx = backbone.begin_frame(history.rx, frame_id)
        target = backbone.aligned_latent(rgb, tx)
        return tx, rx, target

    @torch.no_grad()
    def commit(self, tx, rx, ordered, table, matrix, history):
        prefix = ordered[:, :REF_BUDGET[tx.intra]]
        a = self.codec.recover_ids(prefix, table, tx, matrix)
        b = self.codec.recover_ids(prefix, table, rx, matrix)
        x = tx.commit_reference_once(self.codec.projector(a, tx.intra), a)
        y = rx.commit_reference_once(self.codec.projector(b, rx.intra), b)
        history.submit(x, y)

    @torch.no_grad()
    def burn_in(self, rgb, history, frame_id, table):
        tx, rx, target = self.encode(rgb, history, frame_id)
        matrix = self.codec.transform.matrix(tx.intra)
        ids = self.codec.fsq.encode_ids(self.codec.coefficients(target, matrix), tx.intra)
        self.commit(tx, rx, ids.reshape(1, SLOT_COUNT).index_select(1, table), table, matrix, history)

    def train_frame(self, rgb, history, frame_id: int, table, step: int, length: int) -> dict:
        codec = self.codec
        tx, rx, target = self.encode(rgb, history, frame_id)
        matrix = codec.transform.matrix(tx.intra)
        quantized = codec.fsq.encode_train(codec.coefficients(target, matrix), tx.intra)
        ids = quantized.ids.reshape(1, SLOT_COUNT)
        ordered = ids.index_select(1, table)
        values = quantized.values.reshape(1, SLOT_COUNT, GROUP_WIDTH)
        digits = quantized.digit_position.reshape(1, SLOT_COUNT, GROUP_WIDTH)
        bits = codec.entropy.bits(ids, codec.slot_groups, tx.intra, digits)
        self.commit(tx, rx, ordered, table, matrix.detach(), history)
        with torch.no_grad():
            _, target_ref = codec.backbone.decode_aligned(target, tx)
        crop_rng = torch.Generator().manual_seed(self.seed * 100_000_000 + step * 1000 + frame_id)
        crops = crop_plan(1080, 1920, crop_rng)
        budgets = self.budgets.draw(tx.intra)
        record = self.backward_budgets(rgb, rx, target, target_ref, table, matrix, values, bits, budgets, crops, length)
        return {"frame": frame_id, "frame_type": "I" if tx.intra else "P", "budgets": record,
                "saturation": quantized.saturated.float().mean((0, 2, 3)).cpu().tolist()}

    def backward_budgets(self, rgb, context, target, target_ref, table, matrix, values, bits, budgets, crops, length):
        reference = REF_BUDGET[context.intra]
        codec = self.codec
        weights = self.config["loss"]
        masks = {k: codec.scan.mask(table, k) for k in {*budgets, reference}}
        records = {}
        for index, budget in enumerate(budgets):
            mask = masks[budget]
            latent = codec.recover(values * mask.view(1, -1, 1), mask, context, budget, matrix, checkpointed=True)
            rec, _ = codec.backbone.decode_aligned(latent, context, checkpointed=True)
            coded = bits * masks[max(budget, reference)]
            terms = self.objective(rgb, rec, target, latent, coded, crops)
            weighted = {k: weights[k] * term / len(budgets) for k, term in terms.items()}
            if budget == reference:
                feature = checkpoint(codec.projector, latent, context.intra, use_reentrant=False)
                terms["reference"] = F.l1_loss(feature.float(), target_ref.float())
                weighted["reference"] = weights["reference"] * terms["reference"]
            values_for_log = torch.stack([*terms.values(), *weighted.values(), coded.sum()]).detach().tolist()
            if not all(math.isfinite(v) for v in values_for_log):
                raise FloatingPointError(f"nonfinite Joint objective on frame {context.frame_id}, K{budget}")
            n = len(terms)
            records[str(budget)] = {"terms": dict(zip(terms, values_for_log[:n], strict=True)),
                                    "weighted": dict(zip(weighted, values_for_log[n:-1], strict=True)),
                                    "proxy_bits": values_for_log[-1], "display_tokens": budget,
                                    "transmitted_tokens": max(budget, reference)}
            (sum(weighted.values()) / length).backward(retain_graph=index + 1 < len(budgets))
        return records

    def update(self, clip, window: Window, step: int) -> dict:
        if len(clip) != window.span or self.scheduler.last_epoch != step - 1:
            raise ValueError("window span or optimizer step is out of sequence")
        stage = self.schedule(step)
        codec = self.codec.train()
        table = codec.scan.table()
        history = History.at(window.anchor)
        self.optimizer.zero_grad(set_to_none=True)
        frames = []
        for frame_id, frame in enumerate(clip, window.anchor):
            rgb = frame[None].to(codec.device, dtype=torch.float32).div_(255)
            with torch.autocast(device_type=codec.device.type, dtype=torch.bfloat16, cache_enabled=False):
                if frame_id < window.start:
                    self.burn_in(rgb, history, frame_id, table)
                else:
                    frames.append(self.train_frame(rgb, history, frame_id, table, step, window.length))
        grad = torch.nn.utils.clip_grad_norm_(self.parameters, self.config["clip_norm"], error_if_nonfinite=True)
        lrs = {group["name"]: group["lr"] for group in self.optimizer.param_groups}
        self.optimizer.step()
        self.scheduler.step()
        return {"step": step, "stage": stage, "window": asdict(window), "frames": frames,
                "grad_norm": grad.item(), "lr": lrs}


def _validation(run, seed, step):
    from .checkpoint import isolated_rng
    from .evaluation import evaluate
    from .protocol import OUTPUT

    directory = OUTPUT / f"seed{seed}" / "periodic" / f"step_{step:05d}"
    if step and step % run.config["validate_every"] == 0 and not (directory / "evaluation.json").is_file():
        with isolated_rng():
            evaluate(run, "validation", directory)


def _checkpoint(run, trainer, store, rows, journal, progress):
    from .checkpoint import make_checkpoint

    if run.data.counts["test"]:
        raise RuntimeError("Train and Validation must have zero Test pixel reads")
    run.verify_integrity()
    artifact = make_checkpoint(trainer, rows, journal, run.identity, progress)
    return store.write(artifact, journal)


def _progress(run, step, timings, budget_counts, train_peak=0.0):
    return {"step": step, "stage_seconds": dict(timings), "budget_counts": dict(budget_counts),
            "pixel_reads": dict(run.data.counts), "peak_gpu_mib": train_peak}


def publish_status(run, seed, store, record, progress, status):
    from aip_vtok.diagnostics.diag0.io import json_at
    from aip_vtok.runtime import write_json

    from .protocol import CHECKPOINTS, OUTPUT

    status_path = OUTPUT / f"train_seed{seed}.json"
    data = {"status": status, "steps": progress["step"], "target": run.config["steps"],
            "identity": run.identity, "checkpoint": record, "checkpoint_directory": str(store.directory),
            "provenance": run.provenance["frozen_tensors_sha256"], "config": run.config,
            "parameters": {name: sum(p.numel() for p in module.parameters())
                           for name, module in run.codec.named_children()}, **progress}
    write_json(status_path, data)
    manifest = {f"seed{i}": json_at(path) for i in (1, 2)
                if (path := CHECKPOINTS / f"seed{i}" / "index.json").exists()}
    write_json(OUTPUT / "checkpoint_manifest.json", manifest)


def train_experiment(run, seed: int, *, resume: bool) -> None:
    import signal
    from collections import Counter, defaultdict
    from time import perf_counter

    from .checkpoint import CheckpointStore, Journal, restore_training
    from .evaluation import select_checkpoint
    from .protocol import CHECKPOINTS, OUTPUT

    trainer = Trainer(run.codec, run.config, seed)
    store = CheckpointStore(CHECKPOINTS / f"seed{seed}", run.identity)
    journal_path = OUTPUT / f"seed{seed}" / "updates.jsonl"
    rows = run.data.catalog("train")
    if resume:
        artifact, record = store.load()
        restore_training(trainer, rows, artifact)
        journal = Journal(journal_path, artifact["journal"])
        timings = defaultdict(float, artifact["progress"]["stage_seconds"])
        counts = Counter(artifact["progress"]["budget_counts"])
        run.data.counts.update(artifact["progress"]["pixel_reads"])
    else:
        if store.entries() or journal_path.exists():
            raise FileExistsError("training already has a journal or checkpoint; use --resume")
        journal = Journal(journal_path, None)
        timings, counts = defaultdict(float), Counter()
        record = _checkpoint(run, trainer, store, rows, journal, _progress(run, 0, timings, counts))
    train_peak = artifact["progress"]["peak_gpu_mib"] if resume else 0.0
    stop = False
    request_checkpoint = False

    def save_next(_signum, _frame):
        nonlocal request_checkpoint
        request_checkpoint = True

    def request_stop(_signum, _frame):
        nonlocal stop
        stop = True

    old = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    old[signal.SIGUSR1] = signal.signal(signal.SIGUSR1, save_next)
    try:
        progress = _progress(run, journal.step, timings, counts, train_peak)
        publish_status(run, seed, store, record, progress, "INCOMPLETE")
        _validation(run, seed, journal.step)
        torch.cuda.reset_peak_memory_stats(run.codec.device)
        while not stop and (window := trainer.next_window(rows, journal.step + 1)) is not None:
            if journal.step + 1 == run.config["scan_freeze_step"]:
                export_scan(run.codec.scan, store.directory)
            started = perf_counter()
            clip = run.data.read("train", window.video, window.anchor, window.span)
            update = trainer.update(clip, window, journal.step + 1)
            duration = perf_counter() - started
            update["seconds"] = duration
            timings[update["stage"]] += duration
            counts.update(k for frame in update["frames"] for k in frame["budgets"])
            journal.append(update)
            step = journal.step
            train_peak = max(train_peak, torch.cuda.max_memory_allocated(run.codec.device) / 2**20)
            if step % 10 == 0 or step == 1:
                print(f"seed={seed} step={step}/{run.config['steps']} phase={update['stage']} "
                      f"step_seconds={duration:.3f} grad={update['grad_norm']:.6f}", flush=True)
            if request_checkpoint or stop or step % run.config["save_every"] == 0 or step == run.config["steps"]:
                request_checkpoint = False
                progress = _progress(run, step, timings, counts, train_peak)
                record = _checkpoint(run, trainer, store, rows, journal, progress)
                publish_status(run, seed, store, record, progress, "INCOMPLETE")
            if not stop and step % run.config["validate_every"] == 0:
                _validation(run, seed, step)
                torch.cuda.reset_peak_memory_stats(run.codec.device)
        if not stop and journal.step == run.config["steps"]:
            select_checkpoint(run, seed)
            publish_status(run, seed, store, record, _progress(run, journal.step, timings, counts, train_peak), "TRAINED")
        else:
            if record["step"] != journal.step:
                progress = _progress(run, journal.step, timings, counts, train_peak)
                record = _checkpoint(run, trainer, store, rows, journal, progress)
                publish_status(run, seed, store, record, progress, "INCOMPLETE")
            raise SystemExit(130)
    finally:
        for sig, handler in old.items():
            signal.signal(sig, handler)


def export_scan(scan, directory):
    from aip_vtok.runtime import write_json

    table = scan.freeze().cpu().numpy().astype("<u2").tobytes()
    path = directory / "scan_v1.u16le"
    if path.exists() and path.read_bytes() != table:
        raise RuntimeError("replayed scan freeze differs from the exported scan asset")
    temporary = path.with_suffix(".tmp")
    temporary.write_bytes(table)
    temporary.replace(path)
    write_json(directory / "scan_v1.json", scan.record())
