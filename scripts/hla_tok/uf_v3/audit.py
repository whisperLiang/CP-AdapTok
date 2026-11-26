"""Read-only V2 checkpoint audit and Train-only replay of the original optimization paths."""

import argparse
import hashlib
import os
from pathlib import Path
from time import perf_counter

import torch

from cp_adaptok.metrics import MetricSuite
from hla_tok.evaluation.io import write_json
from hla_tok.uf.official import OfficialAssets, sha256_file
from hla_tok.uf_v2.native_weights import NativeAdjustments, install_adjustments
from hla_tok.uf_v2.runtime import Analysis
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v2 import train as pilot
from scripts.hla_tok.uf_v2 import train_joint as joint
from scripts.hla_tok.uf_v2.history import git, tree
from scripts.hla_tok.v2.protocol import ROOT, csv_at, json_at, setup

BASELINE = "ffbb7177e869b49934bcddf37c24fc7eade8bf41"
SEED_HASHES = {
    1: "13ce52d63722d5fa01d0f780a5113ba36630921d55dd9738c55f4584ec3334d9",
    2: "09bb3a578d89ddb9b2e621a9cf2a1c985edaa49538f236046439116c3864c1ec",
}


def tensor_sha(value):
    data = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8)
    return hashlib.sha256(data.numpy()).hexdigest()


def tensor_delta(before, after):
    diff = before.detach().float().cpu() - after.detach().float().cpu()
    return {"shape": list(before.shape), "before": tensor_sha(before), "after": tensor_sha(after),
            "changed": int(torch.count_nonzero(diff)), "numel": diff.numel(),
            "l2": float(diff.norm()), "max_abs": float(diff.abs().max())}


def checkpoints(root, seed):
    directory = root / "checkpoints/hla_tok/uf_v2"
    return {n: directory / f"d1{suffix}-seed{seed}-step{n}.pt"
            for n, suffix in ((500, ""), (2000, ""), (8000, "-joint"), (20000, "-joint"))}


def inventory(root, assets):
    sources = {"v2": (root, BASELINE), "uf": (root.parents[1] / "hla-tok-uf/CP-AdapTok",
                                              "3efe25ea47308cd7786888f0f5adbd0eb449e64d"),
               "cc": (root.parents[1] / "hla-tok-cc/CP-AdapTok",
                      "a483a212d8efeaeab045daf4f0f5d0830d69928f")}
    manifest = {}
    for name, (path, commit) in sources.items():
        if git(path, "rev-parse", "HEAD") != commit or git(path, "status", "--porcelain", "-uno"):
            raise ValueError(f"{name} worktree identity or tracked files differ")
        weights = {str(item.relative_to(path)): sha256_file(item)
                   for item in sorted((path / "checkpoints/hla_tok").rglob("*.pt"))}
        manifest[name] = {"path": str(path), "commit": commit, "files": tree(path, commit), "checkpoints": weights}
    for seed, expected in SEED_HASHES.items():
        if sha256_file(checkpoints(root, seed)[20000]) != expected:
            raise ValueError(f"V2 seed {seed} checkpoint does not match its ledger")
    official = {str(path.resolve()): sha256_file(path) for path in (assets.intra, assets.ld)}
    build = ROOT / ".build/uf_v3"
    original_builds = sorted(path for part in ("native", "records", "legacy") for path in (build / part).glob("*.so"))
    extensions = {str(path.relative_to(ROOT)): sha256_file(path) for path in original_builds}
    return {"sources": manifest, "official_weights": official, "extensions": extensions}


def checkpoint_groups(state):
    optimizer = state["optimizer"]
    fields = ("refiner", "quantizer", "native_adjustments")
    return [
        {"module": field, "lr": group["lr"], "weight_decay": group["weight_decay"],
         "parameters": {name: {"numel": tensor.numel(),
                               "adam_step": int(optimizer["state"].get(key, {}).get("step", 0))}
                        for (name, tensor), key in zip(state[field].items(), group["params"], strict=True)}}
        for field, group in zip(fields, optimizer["param_groups"], strict=False)
    ]


def archived_training(root, seed):
    paths = checkpoints(root, seed)
    ledger_root = root / "results/hla_tok/uf_v2/training"
    prior = None
    stages = {}
    specifications = (
        ("P1", f"seed{seed}", 1, 2000),
        ("P2", f"joint-seed{seed}", 2001, 8000),
        ("P3", f"formal-seed{seed}", 8001, 20000),
    )
    for phase, folder, first, last in specifications:
        unique = {}
        duplicates = 0
        for path in sorted((ledger_root / folder).glob("steps-*.csv")):
            for row in csv_at(path):
                index = int(row["step"])
                if index in unique:
                    duplicates += 1
                if unique.setdefault(index, row) != row:
                    raise ValueError(f"conflicting archived step {index}")
        rows = [unique[index] for index in sorted(unique)]
        if sorted(unique) != list(range(first, last + 1)):
            raise ValueError(f"step ledger is not contiguous for {seed}/{phase}")
        state = torch.load(paths[last], map_location="cpu", weights_only=True)
        if state["step"] != last:
            raise ValueError("checkpoint step and ledger disagree")
        groups = checkpoint_groups(state)
        expected = last if phase == "P1" else last - 2000
        inter = [entry["adam_step"] for name, entry in groups[0]["parameters"].items() if name.startswith("inter.")]
        if not inter or set(inter) != {expected}:
            raise ValueError(f"unexpected inter AdamW step counters: {seed}/{phase}")
        if phase != "P1":
            counters = {key: value["adam_step"] for key, value in groups[2]["parameters"].items()}
            if counters != {"encoder_logits": last - 2100, "reconstruction_logits": last - 2600,
                            "hyper_logits": last - 2600}:
                raise ValueError(f"unexpected native adapter counters: {seed}/{phase}")
        first_state = (torch.load(paths[500], map_location="cpu", weights_only=True)
                       if prior is None else prior)
        tensors = {}
        for key in ("refiner", "quantizer", "native_adjustments"):
            before = first_state.get(key)
            if before is None and key in state:
                before = {name: torch.zeros_like(value) for name, value in state[key].items()}
            if before is not None:
                tensors[key] = {name: tensor_delta(before[name], value) for name, value in state[key].items()}
        node_files = [path for path in (ledger_root / folder).glob("step-*.json")
                      if first <= int(path.stem.split("-")[-1]) <= last]
        if phase == "P1":
            node_files = [path for path in node_files if path.stem in {"step-500", "step-2000"}]
        nodes = [json_at(path) for path in node_files]
        unroll_key = "sequence_length" if phase == "P1" else "unroll"
        gradient = {key: {"nonzero_steps": sum(float(row[key]) != 0 for row in rows),
                          "max": max(float(row[key]) for row in rows)}
                    for key in rows[0] if key.endswith("gradient_norm")}
        stages[phase] = {
            "checkpoint": str(paths[last]), "sha256": sha256_file(paths[last]), "logged_optimizer_updates": len(rows),
            "groups": groups, "differences": tensors, "comparison_from_step": first_state["step"], "duplicated_log_rows": duplicates,
            "gradients": gradient, "unroll": sorted({int(row[unroll_key]) for row in rows}),
            "frame_instances": sum(int(row[unroll_key]) for row in rows),
            "step_wall_seconds": sum(float(row["step_s"]) for row in rows),
            "gpu_hours_as_logged": sum(node.get("gpu_hours", 0) for node in nodes),
            "cache_seconds_as_logged": sum(node.get("cache_seconds", 0) for node in nodes),
            "teacher_cache_and_image_io_separately_timed": "UNVERIFIED",
            "scope": "feature_space" if phase == "P1" else "feature_space_with_bounded_native_adaptation",
        }
        prior = state
    return stages


def gradient_norm(loss, named, retain_graph=False):
    gradients = torch.autograd.grad(loss, list(named.values()), allow_unused=True, retain_graph=retain_graph)
    return {key: float(grad.detach().float().norm()) if grad is not None else 0.0
            for key, grad in zip(named, gradients, strict=True)}


def replay_pilot(assets, reader, clip, checkpoint, lpips):
    import random

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = joint.UnifiedRefinement().cuda().eval().requires_grad_(False)
    model.load_state_dict(state["quantizer"])
    refiner = joint.SharedRefinement().cuda().to(memory_format=torch.channels_last)
    refiner.load_state_dict(state["refiner"])
    start = perf_counter()
    decoder, (frames,) = pilot.collect_examples(assets, model, reader, [clip], (5, 10, 20, 40), (0, 8))
    prep = perf_counter() - start
    decoder.intra.requires_grad_(False)
    decoder._video().requires_grad_(False)
    selected = frames
    config = json_at(pilot.CONFIG)
    crop = pilot.Crop.sample(random.Random(31), [128, 192], 3)
    tick = perf_counter()
    recorded = pilot.step_batch(refiner, decoder, lpips, torch.optim.AdamW(refiner.parameters()),
                                selected, crop, config)
    torch.cuda.synchronize()
    return {"inputs": {"cached_rgb": list(selected[0].rgb.shape), "cached_feature": list(selected[0].feature.shape),
                       "crop_target": [len(selected) * 4, 3, 128, 192]},
            "train_frames": len(selected), "budget_branches": 4, "data_teacher_cache_wall_s": prep,
            "replayed_optimizer_updates": 1, "update_wall_s": perf_counter() - tick,
            "rate_gradient": "NOT_AVAILABLE", "recorded": recorded}


def replay_joint(assets, reader, clip, checkpoint, lpips):
    import random

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    refiner, model, live, encoded = (joint.SharedRefinement().cuda().to(memory_format=torch.channels_last),
                                     joint.UnifiedRefinement().cuda(), NativeAdjustments().cuda(),
                                     NativeAdjustments().cuda().requires_grad_(False))
    for module, key in ((refiner, "refiner"), (model, "quantizer"), (live, "native_adjustments"),
                        (encoded, "native_adjustments")):
        module.load_state_dict(state[key])
    tick = perf_counter()
    decoder, (frames,) = joint.cache_closed_loop(assets, model, refiner, encoded, reader, [clip])
    prep = perf_counter() - tick
    selected = frames[1:]
    proxy = joint.NativeProxy(decoder, live, encoded)
    crop = pilot.Crop.sample(random.Random(31), [128, 192], 4)
    config = json_at(joint.CONFIG)
    weights = config["weights"]
    named = dict(live.named_parameters()) | {"quantizer." + n: v for n, v in model.named_parameters()}
    named["d1.inter.up.weight"] = refiner.inter.up.weight
    distortion = weights | {"rate": 0.0, "monotonic": 0.0}
    rate = weights | {"rgb_mse": 0.0, "lpips": 0.0, "high_frequency": 0.0, "rate": 1.0, "monotonic": 0.0}
    probes = {}
    for name, terms in (("distortion", distortion), ("rate", rate)):
        loss, _ = joint.losses(refiner, model, proxy, lpips, selected, crop, terms)
        probes[name] = gradient_norm(loss, named)
    optimizer = torch.optim.AdamW([{"params": mod.parameters(), "lr": lr}
                                  for mod, lr in ((refiner, 1e-4), (model, 1e-4), (live, 1e-5))])
    before = {key: value.detach().clone() for key, value in named.items()}
    begin, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    tick = perf_counter()
    begin.record()
    recorded = joint.step((refiner, model, live), proxy, lpips, optimizer, frames, crop, config)
    end.record()
    end.synchronize()
    return {"gradient_l2": probes, "delta": {key: tensor_delta(before[key], value) for key, value in named.items()},
            "inputs": {"cached_rgb": list(selected[0].rgb.shape), "cached_y": list(selected[0].native_y.shape),
                       "cached_context": list(selected[0].ctx.shape), "crop_target_hw": [128, 192]},
            "train_frames": len(frames), "budget_branches": 5, "data_cache_wall_s": prep,
            "replayed_optimizer_updates": 1, "update_cuda_event_ms": begin.elapsed_time(end),
            "update_wall_s": perf_counter() - tick, "recorded": recorded}


@torch.no_grad()
def deployment(assets, reader, clip, checkpoint):
    old, new = Analysis(assets), Analysis(assets)
    adjustments = NativeAdjustments().cuda()
    adjustments.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True)["native_adjustments"])
    before, after = old._video(), new._video()
    install_adjustments(new, adjustments, 8)
    pairs = {
        "encoder.down.weight": (before.encoder.down.weight, after.encoder.down.weight,
                                adjustments.encoder_gain()[:, None, None, None]),
        "encoder.down.bias": (before.encoder.down.bias, after.encoder.down.bias, adjustments.encoder_gain()),
        "recon_head.head.weight": (before.recon_head.head.weight, after.recon_head.head.weight,
                                    adjustments.reconstruction_gain()[:, None, None, None]),
        "recon_head.head.bias": (before.recon_head.head.bias, after.recon_head.head.bias,
                                  adjustments.reconstruction_gain()),
    }
    tensors = {name: tensor_delta(a, b) | {"fold_matches": torch.equal((a.float() * g).half(), b)}
               for name, (a, b, g) in pairs.items()}
    p, q = before.bit_estimator_z.b[8, :, -1], after.bit_estimator_z.b[8, :, -1]
    tensors["bit_estimator_z.b[8,:,-1]"] = tensor_delta(p, q) | {
        "fold_matches": torch.equal((p.float() + adjustments.hyper_bias()).half(), q)}
    observations = []
    for i in range(3):
        rgb = source_frame(reader, clip, i)
        a, b = (endpoint.encode_base(rgb, i, (0, 8), i == 0, False, 9212) for endpoint in (old, new))
        raw_a, raw_b = (endpoint.proxy(i == 0).ufv2_encode_snapshot()[0] for endpoint in (old, new))
        observations.append({"index": i, "kind": "I" if i == 0 else "P", "same_native_payload": a.payload == b.payload,
                             "native_source_shape": list(rgb.shape), "proxy_y_delta": tensor_delta(raw_a, raw_b),
                             "proxy_rgb_delta": tensor_delta(old.preview(a), new.preview(b))})
    if not all(item["fold_matches"] for item in tensors.values()) or not all(
        row["proxy_y_delta"]["changed"] and not row["same_native_payload"] for row in observations[1:]
    ):
        raise ValueError("trained folded weights did not reach CUDA encoding")
    return {"checkpoint_sha256": sha256_file(checkpoint), "fold": tensors, "cuda_frames": observations,
            "probe": "official versus Seed 2 trained fold, fresh FP16 proxies with identical Train input"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify-history", action="store_true")
    args = parser.parse_args()
    setup()
    assets = OfficialAssets.verify(args.weights)
    historical = inventory(args.v2, assets)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "historical_sha256.json"
    if manifest.exists() and json_at(manifest) != historical:
        raise ValueError("an archived input changed")
    if args.verify_history:
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        print("V2, UF and CC worktrees and archived weights verified")
        return
    write_json(manifest, historical)
    stages = {seed: archived_training(args.v2, seed) for seed in (1, 2)}
    reader, plans = inputs()
    clip = plans["fit"][0]
    paths = checkpoints(args.v2, 2)
    lpips = MetricSuite("cuda", ("lpips",)).networks["lpips"].to(memory_format=torch.channels_last)
    replays = {}
    for phase, length, run, node in (("P1", 4, replay_pilot, 2000), ("P2", 8, replay_joint, 8000),
                                      ("P3", 16, replay_joint, 20000)):
        torch.cuda.reset_peak_memory_stats()
        result = run(assets, reader, type(clip)(clip.split, clip.video, clip.start, length), paths[node], lpips)
        replays[phase] = result | {"peak_cuda_bytes": torch.cuda.max_memory_allocated()}
    folded = deployment(assets, reader, clip, paths[20000])
    write_json(args.output / "training_audit.json", {
        "baseline": BASELINE, "seed_stages": stages, "replay": replays, "deployment": folded,
        "native_build": {"torch": torch.__version__, "cuda": torch.version.cuda,
                         "device": torch.cuda.get_device_name(), "omp_wait_policy": os.getenv("OMP_WAIT_POLICY")},
        "read_counts": reader.counts, "test_pixel_reads": reader.counts["test"],
        "finding": "P1 feature-space; P2/P3 bounded P-only adaptation of cached native features and latents",
        "end_to_end": False,
    })


if __name__ == "__main__":
    main()
