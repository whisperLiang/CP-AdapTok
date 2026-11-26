"""Run the R0 RGB-to-RD training and save independently deployable weights."""

import argparse
from dataclasses import dataclass
from pathlib import Path
from random import Random
from time import perf_counter

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_
from torch.utils.checkpoint import checkpoint as recompute

from cp_adaptok.metrics import MetricSuite
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.latent import UnifiedRefinement
from hla_tok.uf.official import OfficialAssets, sha256_file
from hla_tok.uf_v2.decoder import SharedRefinement
from hla_tok.uf_v2.records import BatchRecords
from hla_tok.uf_v2.static_entropy import StaticAsset
from hla_tok.uf_v3.deploy import native_parameters
from hla_tok.uf_v3.joint import SEGMENTS, DifferentiableUF, progressive_frame
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v3.audit import gradient_norm, tensor_delta
from scripts.hla_tok.uf_v3.endpoint import load_v2
from scripts.hla_tok.v2.protocol import ROOT, json_at, setup
from third_party.DCVC.src.models.image_model import DMCI
from third_party.DCVC.src.models.video_model_ld import DMC

CONFIG = ROOT / "configs/hla_tok/uf_v3/joint.json"


def live_model(assets, init):
    stock = load_v2(assets, init)
    model = DifferentiableUF(DMCI(), DMC(), UnifiedRefinement(), SharedRefinement()).cuda()
    model.to(memory_format=torch.channels_last)
    for target, original in ((model.intra, stock.native.intra), (model.inter, stock.native._video()),
                             (model.quantizer, stock.model), (model.refiner, stock.refiner)):
        target.load_state_dict(original.state_dict())
    original = torch.load(init, map_location="cpu", weights_only=True)
    # Preserve unrounded display and quantizer training tensors, while taking
    # the decoded-state backbone baseline directly from the installed V2 fold.
    model.refiner.load_state_dict(original["refiner"])
    model.eval().requires_grad_(False)
    return model, original["native_adjustments"]


def groups(model):
    return {"native_i": native_parameters(model.intra, "native_i"),
            "native_p": native_parameters(model.inter, "native_p"),
            "quantizer": dict(model.quantizer.named_parameters()),
            "refiner": dict(model.refiner.named_parameters())}


def bank_for(model, static=None, raw=False):
    return BatchRecords(tuple(model.quantizer.scales().detach().tolist()), SEGMENTS,
                        static.tables if static else None, raw)


class FitRGB:
    def __init__(self, reader, clips):
        if any(clip.split != "train" for clip in clips):
            raise PermissionError("the training iterator accepts only the frozen Train fit windows")
        self.frames = [torch.stack([reader.read(clip.split, clip.video, clip.start + t)
                                    for t in range(clip.length)]) for clip in clips]
        self.shapes = [list(video.shape) for video in self.frames]

    def sample(self, seed, index, unroll, hw):
        rng = Random((seed << 32) + index)
        video = rng.choice(self.frames)
        start = rng.randrange(len(video) - unroll + 1)
        _, _, native_h, native_w = video.shape
        height, width = hw
        if min(height, width) < 64 or height > native_h or width > native_w:
            raise ValueError("invalid native crop")
        y = 16 * rng.randrange((native_h - height) // 16 + 1)
        x = 16 * rng.randrange((native_w - width) // 16 + 1)
        cut = video[start:start + unroll, :, y:y + height, x:x + width].cuda()
        return cut.float().div_(255).contiguous(memory_format=torch.channels_last)


class Objective:
    def __init__(self, config, static, raw):
        self.static, self.raw = static, raw
        self.weights = config["weights"]
        self.budget_weights = {int(k): v for k, v in config["budget_weights"].items()}
        self.lpips = MetricSuite("cuda", ("lpips",)).networks["lpips"].to(memory_format=torch.channels_last)

    def distortion(self, image, rgb, large):
        mse = (image.float() - rgb.float()).square().mean()
        with torch.autocast("cuda", enabled=False):
            inputs = (image.float() * 2 - 1, rgb.float() * 2 - 1)
            result = recompute(self.lpips, *inputs, use_reentrant=False) if large and image.requires_grad else (
                self.lpips(*inputs))
        return mse, result.mean()

    def distortions(self, predictions, rgb, large):
        by_image = {}
        output = []
        for prediction in predictions:
            key = id(prediction)
            if key not in by_image:
                by_image[key] = self.distortion(prediction, rgb, large)
            output.append(by_image[key])
        mse, lpips = map(torch.stack, zip(*output, strict=True))
        return mse, lpips, self.weights["rgb_mse"] * mse + self.weights["lpips"] * lpips

    @torch.no_grad()
    def target(self, teacher, frames):
        bank = bank_for(teacher)
        state = None
        protection = []
        for rgb in frames.split(1):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = progressive_frame(teacher, bank, rgb, state, budgets=(5, 10))
            state = out.state
            _, _, distortion = self.distortions(out.rgb, rgb, False)
            protection.append(dict(zip(out.budgets[1:], distortion[1:], strict=True)))
        return protection

    def frame(self, output, rgb, reference, large):
        mse, lpips, distortion = self.distortions(output.rgb, rgb, large)
        weights = self.weights
        rate = torch.stack(output.rates)
        mix = distortion.new_tensor([self.budget_weights[b] for b in output.budgets[1:]])
        protect = [torch.relu(d - reference[b] - weights["protection_tolerance"])
                   for b, d in zip(output.budgets, distortion, strict=True) if b in reference]
        base = weights["base"]
        loss_d = base * distortion[0]
        loss_r = base * rate[0]
        if len(output.budgets) > 1:
            loss_d = loss_d + (mix * distortion[1:]).sum() / mix.sum()
            loss_r = loss_r + (mix * rate[1:]).sum() / mix.sum()
            loss_d = loss_d + weights["monotonic"] * (distortion[1:] - distortion[:-1]).relu().mean()
        if protect:
            loss_d = loss_d + weights["protection"] * torch.stack(protect).mean()
        return loss_d, loss_r, (mse.mean(), lpips.mean(), rate.mean())

    def window(self, model, teacher, frames):
        large = frames.shape[-2] >= 720
        tick = perf_counter()
        refs = self.target(teacher, frames)
        torch.cuda.synchronize()
        teacher_wall = perf_counter() - tick
        bank, state = bank_for(model, self.static, self.raw), None
        loss_d, loss_r, metrics, facts = [], [], [], []
        for rgb, ref in zip(frames.split(1), refs, strict=True):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output = progressive_frame(model, bank, rgb, state)
            state = output.state
            d, r, stats = self.frame(output, rgb, ref, large)
            loss_d.append(d)
            loss_r.append(r)
            metrics.append(torch.stack(stats).detach())
            facts.append({"budgets": output.budgets, "base_bytes_estimate": output.estimated_base_bytes,
                          "enhancement_bytes": output.physical_enhancement_bytes})
        return torch.stack(loss_d).mean(), torch.stack(loss_r).mean(), torch.stack(metrics).mean(0), facts, teacher_wall


@dataclass
class Count:
    updates: int = 0

    def after_step(self, *_args, **_kwargs):
        self.updates += 1


def require_raw_rgb(frames: Tensor) -> None:
    if frames.ndim != 4 or frames.shape[1] != 3 or not frames.is_floating_point() or min(frames.shape[-2:]) < 64:
        raise ValueError("training requires consecutive floating-point RGB frames")


def verify_joint_gradients(named, gradients):
    for branch in ("intra_encoder", "inter_encoder"):
        if branch not in named or not named[branch].requires_grad:
            raise RuntimeError(f"{branch} was frozen in the requested joint update")
        for name in ("distortion", "rate"):
            if not gradients[name][branch] > 0.0:
                raise RuntimeError(f"{name} lacks a valid {branch} gradient")


def measure_step(model, teacher, objective, frames, optimizer, parameters, clip, probe, joint):
    require_raw_rgb(frames)
    optimizer.zero_grad(set_to_none=True)
    cuda_first, cuda_last = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    tick = perf_counter()
    cuda_first.record()
    loss_d, loss_r, values, facts, teacher_wall = objective.window(model, teacher, frames)
    gradients = {}
    if probe:
        named = {"intra_encoder": model.intra.enc.enc_2[-1].weight, "inter_encoder": model.inter.encoder.down.weight,
                 "inter_decoder": model.inter.decoder.conv2.weight, "context": next(model.inter.feature_extractor.conv[
                     -1].parameters()),
                 "inter_reconstruction": model.inter.recon_head.head.weight,
                 "inter_entropy": model.inter.bit_estimator_z.b,
                 "inter_quantizer": model.quantizer.inter.logit, "inter_d1": model.refiner.inter.up.weight}
        named = {key: value for key, value in named.items() if value.requires_grad}
        gradients = {"distortion": gradient_norm(loss_d, named, True), "rate": gradient_norm(loss_r, named, True)}
        if joint:
            verify_joint_gradients(named, gradients)
    total = loss_d + objective.weights["rate"] * loss_r
    total.backward()
    norms = {key: float(torch.stack([p.grad.detach().float().norm() for p in group.values() if p.grad is not None]
                                     ).norm()) if any(p.grad is not None for p in group.values()) else 0.0
             for key, group in parameters.items()}
    global_norm = clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    if not float(global_norm) > 0:
        raise RuntimeError("optimizer obtained no effective gradient")
    optimizer.step()
    cuda_last.record()
    cuda_last.synchronize()
    return {"rgb_mse": float(values[0]), "lpips": float(values[1]), "rate_bpp": float(values[2]),
            "loss": float(total.detach()), "norm": float(global_norm), "group_l2": norms, "probes": gradients,
            "input": [list(frames.shape), str(frames.dtype)], "budgets": facts, "wall_seconds": perf_counter() - tick,
            "cuda_interval_ms": cuda_first.elapsed_time(cuda_last), "teacher_wall_seconds": teacher_wall}


def save(path, counts, optimizer, parameters, source, identity):
    dest = path.with_suffix(".tmp")
    torch.save({key: {n: p.detach().cpu() for n, p in group.items()} for key, group in parameters.items()} |
               {"native_adjustments": source, "optimizer": optimizer.state_dict(), "step": counts.updates,
                "signature": identity, "cpu_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, dest)
    dest.replace(path)


def restore(path, parameters, optimizer, identity):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["signature"] != identity:
        raise ValueError("Train, prior, route, seed or code identity changed")
    for name, group in parameters.items():
        with torch.no_grad():
            for key, value in group.items():
                value.copy_(state[name][key])
    optimizer.load_state_dict(state["optimizer"])
    torch.set_rng_state(state["cpu_rng"])
    torch.cuda.set_rng_state(state["cuda_rng"])
    return Count(state["step"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--init", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--entropy", type=Path, required=True)
    parser.add_argument("--admission", type=Path)
    parser.add_argument("--seed", type=int, choices=(1, 2), default=1)
    parser.add_argument("--stage", choices=("P1", "P2", "P3", "P4", "probe"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    args = parser.parse_args()
    decision, config = json_at(args.decision), json_at(CONFIG)
    if decision["selected_route"] != "R0":
        raise ValueError("the fixed-history diagnosis has not selected R0")
    if args.stage == "probe" and args.seed != 1:
        raise ValueError("the limited native joint probe is registered only for Seed 1")
    if args.stage != "probe" and args.steps > config["max_steps"][args.stage]:
        raise ValueError("this stage exceeds its frozen cumulative update cap")
    if args.seed == 2 and (not args.admission or json_at(args.admission)["seed2"] != "PASS"):
        raise ValueError("Seed 2 requires a passing frozen Seed 1 admission decision")
    entropy = json_at(args.entropy)
    backend = entropy["default"]
    static = StaticAsset.load(args.entropy.with_name("static_cdf.json")) if backend != "E0" else None
    setup(args.seed)
    assets = OfficialAssets.verify(args.weights)
    start = perf_counter()
    reader, plans = inputs()
    data = FitRGB(reader, plans["fit"])
    data_seconds = perf_counter() - start
    model, source = live_model(assets, args.init)
    teacher, _ = live_model(assets, args.init)
    parameters = groups(model)
    joint = args.stage in {"P3", "P4", "probe"}
    for key, group in parameters.items():
        for value in group.values():
            value.requires_grad_(joint or not key.startswith("native"))
    optimizer = torch.optim.AdamW([
        {"params": list(group.values()), "lr": config["backbone_lr" if key.startswith("native") else "display_lr"]}
        for key, group in parameters.items()], weight_decay=config["weight_decay"], fused=True)
    identity = {"seed": args.seed, "init": sha256_file(args.init), "config": sha256_file(CONFIG), "data": reader.hashes,
                "decision": sha256_file(args.decision), "train": sha256_file(Path(__file__)),
                "graph": sha256_file(ROOT / "hla_tok/uf_v3/joint.py"), "native_link": decision["native_link"],
                "entropy": backend, "static_cdf": static.digest if static else None}
    count = restore(args.resume, parameters, optimizer, identity) if args.resume else Count()
    first = count.updates
    if args.steps <= first:
        raise ValueError("the requested finish must be beyond the previous optimizer step")
    if args.stage == "probe" and args.steps - first > config["bounded_joint_probe"]["max_optimizer_updates"]:
        raise ValueError("the finite native probe exceeded its frozen update cap")
    watch = optimizer.register_step_post_hook(count.after_step)
    args.checkpoints.mkdir(parents=True, exist_ok=True)
    hw = tuple(config["crop_hw"])
    unroll = config["unroll"].get(args.stage, 4)
    objective = Objective(config, static, backend == "E2")
    before = {key: {name: param.detach().cpu().clone() for name, param in group.items()}
              for key, group in parameters.items()}
    timings, detailed = [], []
    torch.cuda.reset_peak_memory_stats()
    update_start = perf_counter()
    print(args.stage, first, args.steps, backend, flush=True)
    for index in range(first, args.steps):
        native = args.stage == "probe" and index == first
        if native:
            model.inter.set_use_ckpt(True)
        frames = data.sample(args.seed, index, unroll, (1080, 1920) if native else hw)
        metrics = measure_step(model, teacher, objective, frames, optimizer, parameters, config["clip"],
                               probe=index == first, joint=joint)
        if native:
            model.inter.set_use_ckpt(False)
        if native or index == first or count.updates % 25 == 0:
            detailed.append({"step": count.updates, **metrics})
            print(count.updates, metrics["loss"], metrics["group_l2"], flush=True)
        timings.append({"step": count.updates, "unroll": unroll, "native": native,
                        **{k: v for k, v in metrics.items() if isinstance(v, float)}})
        if count.updates % 500 == 0 or count.updates == args.steps:
            path = args.checkpoints / f"r0-{args.stage}-seed{args.seed}-step{count.updates}.pt"
            save(path, count, optimizer, parameters, source, identity)
    watch.remove()
    if count.updates != args.steps:
        raise ValueError("optimizer counter and target disagree")
    delta = {key: {name: tensor_delta(before[key][name], value) for name, value in group.items()}
             for key, group in parameters.items()}
    write_csv(args.output / "steps.csv", timings)
    write_json(args.output / "summary.json", {"status": "PASS", "stage": args.stage, "first_step": first,
        "optimizer_updates": count.updates, "added": count.updates - first, "seed": args.seed,
        "delta": delta, "detail": detailed, "checkpoint": str(path), "checkpoint_sha256": sha256_file(path),
        "gpu_event_interval_seconds": sum(row["cuda_interval_ms"] for row in timings) / 1000,
        "wall_seconds": perf_counter() - start, "update_wall_seconds": perf_counter() - update_start,
        "raw_rgb_cache_wall_seconds": data_seconds, "teacher_wall_seconds": sum(
            row["teacher_wall_seconds"] for row in timings), "fit_rgb_shapes": data.shapes, "read_counts": reader.counts,
        "peak_cuda_bytes": torch.cuda.max_memory_allocated(), "frame_instances": unroll * (count.updates - first),
        "resume": str(args.resume) if args.resume else None, "identity": identity,
        "rate_scope": "differentiable Base bytes estimate plus actual enhancement record packing; physical evaluation separate"})


if __name__ == "__main__":
    main()
