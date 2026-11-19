"""Synchronized native per-frame timing on the real receiver/codec path."""

import math
import time
from itertools import product

import numpy as np
import torch
from torch import nn

from aip_vtok.c1.frames import capture
from aip_vtok.c1.scan import BUDGETS, GROUP_WIDTH, GROUPS, REF_BUDGET, SHAPE, gaussian_kernel
from aip_vtok.diagnostics.diag0.io import json_at
from aip_vtok.runtime import file_hash, write_json
from aip_vtok.sparse_rfsq.rollout import CausalPair

from .phase1 import architecture_counts
from .pilot import METHODS, completed_training, receivers
from .protocol import OUTPUT, PROVENANCE, SIGMA, gpu_snapshot

WARMUP, REPEATS = 5, 30
STAGES = ("backbone_encode", "token_select_quantize", "receiver_context", "receiver_recovery",
          "rgb_decoder", "reference_projection_and_commit")
LATENCY = OUTPUT / "pilot_latency.json"


def timing_stats(values):
    return {"mean_ms": float(np.mean(values)), "p50_ms": float(np.percentile(values, 50)),
            "p95_ms": float(np.percentile(values, 95))}


def union_budget(k, intra):
    return max(k, REF_BUDGET[intra])


@torch.no_grad()
def timed_frame(codec, prior, image, frame, case, receiver, k):
    backbone = codec.backbone
    events = [torch.cuda.Event(enable_timing=True) for _ in range(len(STAGES) + 1)]
    pair = CausalPair(prior.tx, prior.rx)
    torch.cuda.synchronize()
    start = time.perf_counter()
    events[0].record()
    sample = capture(backbone, pair.tx, frame, image)
    events[1].record()
    intra = sample.context.intra
    ids, _ = case.layout.encode(sample)
    packet = case.plan.encode_prefix(ids, union_budget(k, intra))
    events[2].record()
    ctx = backbone.begin_frame(pair.rx, frame)
    events[3].record()
    latent = receiver(case, backbone, ctx, packet[:, :k])
    events[4].record()
    rgb, _ = backbone.decode_aligned(latent, ctx)
    events[5].record()
    ref_budget = REF_BUDGET[intra]
    ref = packet[:, :ref_budget]
    tx_latent = receiver(case, backbone, sample.context, ref)
    rx_latent = latent if k == ref_budget else receiver(case, backbone, ctx, ref)
    pair.commit(sample.context, tx_latent, codec.projector(tx_latent, intra), ctx,
                rx_latent, codec.projector(rx_latent, intra))
    events[6].record()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) * 1000
    if rgb.shape != (1, 3, 1080, 1920) or pair.rx.frame_id != frame or not ctx.committed:
        raise RuntimeError("timed frame must use native RGB and a synchronized single commit")
    metrics = {name: a.elapsed_time(b) for name, a, b in zip(STAGES, events[:-1], events[1:], strict=True)}
    return metrics | {"end_to_end": elapsed, "cuda_span": events[0].elapsed_time(events[-1])}


def estimate_macs(receiver, codec, case, frame):
    counts = []

    def conv(layer, inputs, output):
        counts.append(output.numel() * math.prod(layer.kernel_size) * layer.in_channels // layer.groups)

    model = receiver.model
    hooks = [layer.register_forward_hook(conv) for layer in model.modules()
             if isinstance(layer, nn.Conv2d)] if model is not None else []
    try:
        size = len(gaussian_kernel(SIGMA, frame.payload.device, frame.sample.aligned.dtype))
        budget = REF_BUDGET[frame.context.intra]
        receiver(case, codec.backbone, frame.context, frame.payload[:, :budget])
    finally:
        for hook in hooks:
            hook.remove()
    gaussian = 2 * math.prod(SHAPE) * GROUPS * (GROUP_WIDTH + 1) * size
    macs = sum(counts) + gaussian
    return {"network_macs": sum(counts), "gaussian_macs": gaussian, "recovery_macs": macs,
            "recovery_flops_estimate": 2 * macs}


@torch.no_grad()
def measure_arm(codec, clip, cases, label, receiver, count):
    intra_prior = CausalPair()
    prior_p = CausalPair()
    frame_i = prior_p.advance(codec, clip[0], 0, cases, receiver)
    frame_p = CausalPair(prior_p.tx, prior_p.rx).advance(codec, clip[1], 1, cases, receiver)
    for frame, k in product((frame_i, frame_p), BUDGETS):
        index = frame.context.frame_id
        intra = frame.context.intra
        kind = frame.sample.frame_type
        case = cases[kind]
        prior = intra_prior if intra else prior_p
        complexity = estimate_macs(receiver, codec, case, frame)
        for _ in range(WARMUP):
            timed_frame(codec, prior, clip[index], index, case, receiver, k)
        torch.cuda.reset_peak_memory_stats()
        resident = torch.cuda.memory_allocated()
        sampled = [timed_frame(codec, prior, clip[index], index, case, receiver, k) for _ in range(REPEATS)]
        peak = torch.cuda.max_memory_allocated()
        ref = REF_BUDGET[intra]
        yield {"arm": label, "network": METHODS[label], "frame_type": kind, "budget": k, "parameters": count,
               **complexity, "timing": {m: timing_stats([r[m] for r in sampled]) for m in sampled[0]},
               "peak_gpu_allocated_mib": peak / 2**20, "incremental_peak_gpu_allocated_mib": (peak - resident) / 2**20,
               "rate": {"display_budget": k, "reference_budget": ref, "transmitted_union_symbols": union_budget(k, intra),
                        "display": case.rates[k], "reference": case.rates[ref],
                        "required_union": case.rates[union_budget(k, intra)]}}


def run_latency(run):
    training = completed_training()
    models = receivers(run, training)
    before = gpu_snapshot()
    checkpoint_hashes = {arm: record["checkpoints_sha256"]["500"] for arm, record in training.items()}
    if LATENCY.is_file():
        existing = json_at(LATENCY)
        if (existing["checkpoints_sha256"] != checkpoint_hashes
                or existing["provenance_sha256"] != file_hash(PROVENANCE) or not existing["integrity"]["passed"]):
            raise RuntimeError("stored latency measurements use different inputs")
        print("Latency already verified.", flush=True)
        return
    catalog = run.reader.open("train")
    holdout = run.lock["holdout"][0]
    clip = run.reader.read(catalog, holdout["video"], holdout["start"], 2)
    cases = run.inputs.cases("SP-1", run.lock["keys"]["SP-1"])
    counts = architecture_counts(run.w1)
    rows = []
    for arm, receiver in models.items():
        rows.extend(measure_arm(run.codec, clip, cases, arm, receiver, counts[arm]))
        print(f"Latency · {arm} · I/P × 5 budgets · {REPEATS} synchronized repeats", flush=True)
    after = gpu_snapshot()
    write_json(LATENCY, {
        "hardware": torch.cuda.get_device_name(0), "gpu_start": before, "gpu_end": after,
        "batch": 1, "rgb_height_width": [1080, 1920], "precision": "FP32", "warmup": WARMUP, "repeats": REPEATS,
        "device": "cuda:0", "source": {"split": "train", "video": holdout["video"], "start": holdout["start"],
                                       "rgb_sha256": [item["sha256"] for item in
                                                      catalog.rows[holdout["video"]]["frame_hashes"]
                                                      [holdout["start"]:holdout["start"] + 2]]},
        "timing_method": "CUDA events for phases. Synchronized wall time for encode+decode, both reference "
                         "projections and commits. The sender and receiver histories are independent. "
                         "The receiver reuses display recovery when its budget equals the reference budget.",
        "complexity_method": "Count learned Conv2d and the two normalized Gaussian convolutions. FLOPs = 2 × MACs. "
                             "Timing also includes masks, FSQ decoding, layout/scatter, activation, pool, upsample, "
                             "fusion and hard observation preservation.",
        "rate_method": "Fixed 16-bit symbols, declared packet framing bytes, and Train marginal entropy estimates. "
                       "The union is the larger of the two nested prefixes.",
        "provenance_sha256": file_hash(PROVENANCE), "checkpoints_sha256": checkpoint_hashes,
        "integrity": run.verify_integrity(), "rows": rows,
    })
