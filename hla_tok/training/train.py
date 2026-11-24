"""The Phase 1 continuous diagnosis and quantized fixed-budget curriculum."""

import json
import random
import signal
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from time import perf_counter

import torch

from hla_tok import Geometry
from hla_tok.evaluation.benchmark import transfer
from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.evaluation.reconstruction import validate
from hla_tok.training.checkpoint import read_checkpoint, save_checkpoint
from hla_tok.training.step import initialize_codebook, optimize_batch


def stage_schedule(config, stage):
    return [Geometry(leg["height"], leg["width"]) for leg in config[stage] for _ in range(leg["steps"])]

def eligible_sources(sources, geometry):
    width, height = geometry.width, geometry.height
    if (height, width) == (1080, 1920):
        result = [row for row in sources if row["resolution"] == [width, height]]
    else:
        result = [row for row in sources if row["resolution"][0] >= width and row["resolution"][1] >= height]
    if not result:
        raise ValueError(f"no source Train frames can supply {(height, width)}")
    return result


class TrainingFrames:
    def __init__(self, reader, schedule, seed):
        self.reader, self.schedule, self.seed = reader, schedule, seed
        sources = [row for row in reader.catalog("train").values()
                   if row["official_split"] == row["role"] == "train" and not row["resized"]]
        self.catalog = {geometry: eligible_sources(sources, geometry) for geometry in dict.fromkeys(schedule)}

    def read(self, step):
        rng = random.Random(self.seed + step)
        geometry = self.schedule[step]
        row = rng.choice(self.catalog[geometry])
        index = rng.randrange(len(row["frame_hashes"]))
        w, h = row["resolution"]
        y = rng.randrange(h - geometry.height + 1)
        x = rng.randrange(w - geometry.width + 1)
        raw = self.reader.read("train", row["content_id"], index)
        crop = raw[:, y:y+geometry.height, x:x+geometry.width]
        flip = bool(rng.randrange(2))
        if flip:
            crop = crop.flip(-1)
        perceptual = (rng.randrange(max(1, geometry.height - 255)), rng.randrange(max(1, geometry.width - 255)))
        sample = {"video": row["content_id"], "frame": index, "crop_xyhw": [x, y, geometry.width, geometry.height],
                  "source_sha256": row["frame_hashes"][index]["sha256"],
                  "flip_horizontal": flip, "perceptual_crop_yx": list(perceptual)}
        return crop, geometry, perceptual, sample

    def batches(self, first):
        # Only one pending reader advances counters. Plans are keyed by completed update index.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(self.read, first)
            for step in range(first + 1, len(self.schedule)):
                current = pending.result()
                pending = pool.submit(self.read, step)
                yield current
            yield pending.result()


@torch.no_grad()
def calibrate(model, reader, count, native_budget, seed, output):
    geometry = Geometry(256, 256)
    source = TrainingFrames(reader, [geometry] * count, seed + 100_000)
    samples, points = [], []
    for raw, _, _, sample in source.batches(0):
        samples.append(sample)
        tokens = model.encode_training(transfer(raw[None]), geometry.scale_budget(native_budget), quantize=False)
        points.append(tokens.values.flatten(0, 1).cpu())
    points = torch.cat(points)
    initialize_codebook(model.quantizer, points, seed=seed)
    write_json(output / "calibration.json", {"frames": samples, "observations": len(points), "lloyd_iterations": 4})


def restore_start(model, optimizer, scheduler, budget_rng, identity, resume, initialize_from):
    if resume is not None:
        saved = read_checkpoint(resume, identity).restore(model, optimizer, scheduler, budget_rng)
        return saved["step"], saved["counters"]
    if initialize_from is not None:
        origin = read_checkpoint(initialize_from).state
        if origin["identity"]["stage"] != "continuous" or origin["step"] != 500:
            raise ValueError("fixed VQ training must start from a completed continuous diagnostic")
        if origin["identity"]["dataset_sha256"] != identity.dataset_sha256:
            raise ValueError("continuous initializer was trained on a different manifest")
        model.load_state_dict(origin["model"])
    return 0, {"processed_pixels": 0, "processed_frames": 0, "budget_exposures": {},
               "optimization_wall_s": 0.0, "latest_codebook_statistics": None}


def append_log(path, row):
    with path.open("a") as stream:
        stream.write(json.dumps(row, allow_nan=False) + "\n")


@contextmanager
def stopping_signal():
    pending_stop = False

    def request_stop(*_):
        nonlocal pending_stop
        pending_stop = True

    handlers = {sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        yield lambda: pending_stop
    finally:
        for sig, previous in handlers.items():
            signal.signal(sig, previous)


def update(model, batch, optimizer, scheduler, losses, config, counters, log, step, *, quantize, bf16, audit):
    raw, geometry, perceptual, sample = batch
    budget = config["native_budget"]
    model.train()
    clock = perf_counter()
    row = optimize_batch(model, transfer(raw[None]), (geometry.scale_budget(budget),), optimizer, losses,
                         quantize=quantize, bf16=bf16, audit=audit, crop_position=perceptual,
                         clip_norm=config["gradient_clip_norm"])
    lr = optimizer.param_groups[0]["lr"]
    scheduler.step()
    seconds = perf_counter() - clock
    counters["processed_frames"] += 1
    counters["processed_pixels"] += geometry.height * geometry.width
    counters["optimization_wall_s"] += seconds
    exposures = counters["budget_exposures"]
    exposures[str(budget)] = exposures.get(str(budget), 0) + 1
    counters["latest_codebook_statistics"] = row["codebook"]
    append_log(log, {"step": step, "sample": sample, "native_budget": budget, "learning_rate": lr,
                     "update_s": seconds} | row)
    if step % 50 == 0:
        print(f"Phase 1 {'fixed' if quantize else 'continuous'} step {step} RGB L1 {row['losses']['rgb_l1']['raw']:.5f}",
              flush=True)


def train(model, reader, cohort, quality, losses, config, identity, output, checkpoint_dir, *,
          resume=None, initialize_from=None):
    started = perf_counter()
    stage = identity.stage
    budget = config["native_budget"]
    seed = config["seed"]
    bf16 = config["precision"] == "bf16"
    quantize = stage == "fixed"
    schedule = stage_schedule(config, stage)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    warmup = config["warmup_steps"]
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: min(1.0, (step + 1) / max(1, warmup)))
    budget_rng = torch.Generator().manual_seed(seed)
    start, counters = restore_start(model, optimizer, scheduler, budget_rng, identity, resume, initialize_from)
    output.mkdir(parents=True, exist_ok=True)
    if start >= len(schedule):
        raise ValueError("the selected checkpoint already finished this curriculum")
    if quantize and start == 0:
        model.eval()
        calibrate(model, reader, config["codebook_calibration_frames"], budget, seed, output)
    log = output / f"updates_after_{start:05d}.jsonl"
    log.touch(exist_ok=False)
    frames = TrainingFrames(reader, schedule, seed)

    completed = start

    def store(step):
        checkpoint = checkpoint_dir / f"step_{step:05d}.pt"
        save_checkpoint(checkpoint, identity, model, optimizer, scheduler, budget_rng, step=step,
                        data_cursor=step, counters=counters)
        write_json(output / "latest_checkpoint.json", {"step": step, "sha256": sha256_file(checkpoint),
                                                       "file": str(checkpoint)})

    with stopping_signal() as stopping:
        try:
            for step, batch in enumerate(frames.batches(start), start=start + 1):
                geometry = batch[1]
                audit = step == start + 1 or schedule[step-2] != geometry
                update(model, batch, optimizer, scheduler, losses, config, counters, log, step,
                       quantize=quantize, bf16=bf16, audit=audit)
                completed = step
                scheduled = step % config["save_every"] == 0
                if scheduled or stopping():
                    store(step)
                if stopping():
                    break
                if scheduled:
                    validate(model, reader, cohort, quality, (budget,), output / "periodic" / f"step_{step:05d}",
                             quantize=quantize, subset=True, bf16=bf16)
        finally:
            write_json(output / "training_progress.json", {"completed_updates": completed, "physical_batch": 1,
                       "effective_batch": 1, "stopping_requested": stopping(),
                       "segment_wall_s_including_periodic_validation": perf_counter() - started} | counters)
        if not stopping():
            validate(model, reader, cohort, quality, (budget,), output, quantize=quantize, bf16=bf16)


def fixed_quality_gate(hla, baseline_table):
    points = sorted((r for r in baseline_table if r["method"] == "Original AdapTok"
                     and str(r["stride"]) == "128" and r["precision"] == "fp32"), key=lambda r: float(r["actual_bpp"]))
    rate = hla["actual_bpp"]
    bracket = next(((lo, hi) for lo, hi in zip(points, points[1:])
                    if float(lo["actual_bpp"]) <= rate <= float(hi["actual_bpp"])), None)
    if bracket is None:
        raise ValueError("observed baseline rates do not bracket the fixed-budget payload")
    lo, hi = bracket
    fraction = (rate - float(lo["actual_bpp"])) / (float(hi["actual_bpp"]) - float(lo["actual_bpp"]))
    match = {k: (1 - fraction) * float(lo[k]) + fraction * float(hi[k]) for k in ("global_psnr", "lpips", "dists")}
    limits = {"lpips_upper": 1.10 * match["lpips"], "dists_upper": 1.10 * match["dists"],
              "global_psnr_lower": match["global_psnr"] - 1.0}
    passes = {"lpips": hla["lpips"] <= limits["lpips_upper"], "dists": hla["dists"] <= limits["dists_upper"],
              "psnr": hla["global_psnr"] >= limits["global_psnr_lower"]}
    return {"passes": passes, "quality_pass": all(passes.values()), "linear_bpp_interpolation": match,
            "thresholds": limits, "observed": hla, "bracket": [float(lo["actual_bpp"]), float(hi["actual_bpp"])]}
