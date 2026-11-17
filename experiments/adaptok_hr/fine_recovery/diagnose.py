"""Frozen-weight, same-real-history representation interventions (diagnostic only)."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from adaptok_hr.codec.codec import Codec
from adaptok_hr.data.dataset import read_clip
from adaptok_hr.data.manifest import load_index
from adaptok_hr.evaluation.evaluate import histogram_summary, metric_manifest, save_image
from adaptok_hr.models.temporal_state import frame_context
from adaptok_hr.runtime import environment, file_hash, seed_everything, write_json
from adaptok_hr.training.checkpoint import load_model
from adaptok_hr.training.train import append_row
from cp_adaptok.metrics import MetricSuite

LEVELS = ("Base", "Medium", "Fine")


def feature_statistics(values):
    x = values.detach().float()
    z = F.normalize(x, dim=-1)
    count = x.shape[1]
    cosine = ((z.sum(1).square().sum(-1) - z.square().sum((1, 2))) /
              max(count * (count - 1), 1)).mean()
    return {"mean": float(x.mean()), "std": float(x.std(unbiased=False)),
            "spatial_variance": float(x.var(1, unbiased=False).mean()),
            "normalized_variance": float(z.var(1, unbiased=False).mean()),
            "mean_position_cosine": float(cosine)}


def highpass(rgb):
    return rgb.float() - F.avg_pool2d(F.pad(rgb.float(), (1, 1, 1, 1), mode="reflect"), 3, 1)


def detail_metrics(rgb, target):
    dx = (rgb[..., 1:] - rgb[..., :-1]) - (target[..., 1:] - target[..., :-1])
    dy = (rgb[..., 1:, :] - rgb[..., :-1, :]) - (target[..., 1:, :] - target[..., :-1, :])
    hp = highpass(rgb) - highpass(target)
    return {"edge_mse": float((dx.square().mean() + dy.square().mean()) * .5),
            "high_frequency_l1": float(hp.abs().mean()), "high_frequency_mse": float(hp.square().mean())}


def interventions(z, discrete, second, budget, fine_offset):
    z, q, other = z[:, :budget], discrete[:, :budget], second[:, :budget]
    fine = q[:, fine_offset:]
    rng = torch.Generator().manual_seed(913)
    order = torch.randperm(fine.shape[1], generator=rng).to(q.device)
    def mixed(f):
        return torch.cat((q[:, :fine_offset], f), 1)
    return {"vq": q, "continuous": z, "fine_continuous": mixed(z[:, fine_offset:]),
            "fine_constant": mixed(fine[:, :1].expand_as(fine)),
            "fine_other_ids": mixed(other[:, fine_offset:]),
            "fine_shuffle": mixed(fine[:, order]), "no_fine": q[:, :fine_offset]}


def nearest_two(z, weight):
    codes = F.normalize(weight.detach().float(), dim=-1)
    scores, ids = zip(*[(chunk @ codes.T).topk(2, dim=-1) for chunk in z.flatten(0, 1).split(512)])
    return (2 - 2 * torch.cat(scores)).reshape(*z.shape[:-1], 2), torch.cat(ids).reshape(*z.shape[:-1], 2)


def layer_slices(model, count):
    offset = 0
    for name, capacity in zip(LEVELS, model.positions.counts):
        end = min(count, offset + capacity)
        if end > offset:
            yield name, slice(offset, end)
        offset += capacity


def gradients(model, z, discrete, reference, target, budget):
    result = {}
    for label, values in (("continuous_token", z), ("decoder_discrete_input", discrete)):
        leaf = values[:, :budget].detach().clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            rgb, _ = model.decoder(leaf, reference)
        gradient, = torch.autograd.grad(F.mse_loss(rgb.float(), target), leaf)
        result[label] = {name: float(gradient[:, span].norm()) for name, span in layer_slices(model, budget)}
    return result


def projection_tokens(model, frame, state):
    raw = []
    handles = [layer.register_forward_hook(lambda module, args, output: raw.append(output.detach()))
               for layer in model.tokenizer.projections]
    try:
        encoded = model.encode_features(frame, state)
    finally:
        for handle in handles:
            handle.remove()
    ordered = torch.cat([x.flatten(2).transpose(1, 2)[:, model.positions.order(level)]
                         for level, x in enumerate(raw)], 1).float()
    return encoded, ordered


def representation_rows(model, raw, z, ids, distances, previous, metadata):
    for count in (sum(model.positions.counts), *model.config.budgets):
        scope = "all_candidates" if count == sum(model.positions.counts) else f"active_K{count}"
        for name, span in layer_slices(model, count):
            d = distances[:, span]
            row = {**metadata, "scope": scope, "level": name, "tokens": span.stop - span.start,
                   "raw": feature_statistics(raw[:, span]), "normalized": feature_statistics(z[:, span]),
                   "nearest_distance": float(d[..., 0].mean()), "second_distance": float(d[..., 1].mean()),
                   "margin": float((d[..., 1] - d[..., 0]).mean()),
                   "histogram": torch.bincount(ids[:, span].flatten(), minlength=8192).cpu().tolist()}
            if previous is not None:
                row["same_position_temporal_mse"] = float((z[:, span] - previous[:, span]).square().mean())
            yield row


def summarize_representations(rows):
    groups = {}
    for row in rows:
        key = f"{row['split']}/{row['scope']}/{row['level']}"
        groups.setdefault(key, []).append(row)
    summary = {}
    for key, group in groups.items():
        counts = torch.tensor([r["histogram"] for r in group]).sum(0)
        summary[key] = {**histogram_summary(counts.tolist()), "observations": len(group),
                        "maximum_code_fraction": float(counts.max() / counts.sum()),
                        "raw_spatial_variance": sum(r["raw"]["spatial_variance"] for r in group) / len(group),
                        "normalized_variance": sum(r["normalized"]["spatial_variance"] for r in group) / len(group),
                        "mean_position_cosine": sum(r["normalized"]["mean_position_cosine"] for r in group) / len(group),
                        "nearest_margin": sum(r["margin"] for r in group) / len(group)}
    return summary


def summarize_bypass(rows):
    groups = {}
    for row in rows:
        groups.setdefault(f"{row['split']}/{row['variant']}", []).append(row)
    names = ("mse", "psnr", "lpips", "dists", "edge_mse", "high_frequency_l1", "output_change_mse")
    return {key: {name: sum(r[name] for r in group) / len(group) for name in names}
            for key, group in groups.items()}


def write_report(output, summary):
    fine = summary["representations"]["validation/active_K2048/Fine"]
    b = summary["bypass"]
    text = ["# E0 冻结权重诊断", "", "本文件由原始 JSON 自动生成；全部旁路为 diagnostic_only，不是可传输率失真结果。", "",
            f"验证集 K2048 Fine 使用 {fine['used_codes']} 个码字，最大占比 {fine['maximum_code_fraction']:.8f}。",
            f"量化前归一化空间方差 {fine['normalized_variance']:.9g}，位置余弦 {fine['mean_position_cosine']:.9g}。", "",
            "| 同历史干预 | MSE | LPIPS | DISTS | 输出变化 MSE |", "|---|---:|---:|---:|---:|"]
    for key, value in b.items():
        if key.startswith("validation/"):
            text.append(f"| {key} | {value['mse']:.9g} | {value['lpips']:.7g} | {value['dists']:.7g} | {value['output_change_mse']:.9g} |")
    text += ["", "确定事实：连续特征存在非零变化不等于它携带可用细节；用旁路损失、离散置换和梯度分别判断。",
             "推测边界：上述冻结模型干预不能单独证明塌缩的训练因果；预算损失范围和共享码本竞争需要匹配的 2×2 实验验证。",
             "Decoder 输入梯度与量化前 Token 梯度（RGB MSE）见 gradients.json；非零梯度只证明局部敏感，不证明质量增益。"]
    (output / "root_cause.md").write_text("\n".join(text) + "\n")


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "summary.json").exists() or (output / "representations.jsonl").exists():
        raise ValueError("diagnostic output exists; use a new experiment directory")
    seed_everything(20261118)
    torch.set_num_threads(4)
    model, checkpoint = load_model(args.checkpoint)
    model.requires_grad_(False)
    index = load_index(args.index)
    metrics = MetricSuite("cuda", ("psnr", "lpips", "dists"))
    if metrics.errors:
        raise RuntimeError(metrics.errors)
    codec = Codec(model, amp_bf16=True)
    representations, bypass, gradient_rows, inputs = [], [], [], []
    for split, ids in (("train", ["DAVIS2017:bear"]),
                       ("validation", ["DAVIS2017:blackswan", "DAVIS2017:car-roundabout", "DAVIS2017:dog"])):
        for source in (r for r in index["splits"][split] if r["content_id"] in ids):
            clip, metadata = read_clip(source, 0, min(args.frames, source["original_frames"]))
            inputs.append(metadata)
            state, previous = None, None
            for frame_index, cpu in enumerate(clip):
                frame = cpu[None].cuda()
                if frame_index in (0, 1, 7, 31, 32, 63):
                    info = {"split": split, "content_id": source["content_id"], "frame": frame_index,
                            "preceding_state_chain": state.chain.hex() if state is not None else None}
                    _, _, reference = frame_context(state, model.config.gop)
                    with torch.no_grad(), codec.autocast():
                        encoded, raw = projection_tokens(model, frame, state)
                    z = F.normalize(raw, dim=-1)
                    distances, nearest = nearest_two(z, model.vq.codebook.weight)
                    discrete = model.vq.lookup(encoded.ids).detach()
                    second = model.vq.lookup(nearest[..., 1]).detach()
                    for row in representation_rows(model, raw, z, encoded.ids, distances, previous, info):
                        representations.append(row)
                        append_row(output / "representations.jsonl", row)
                    previous = z
                    variants = interventions(z, discrete, second, 2048, sum(model.positions.counts[:2]))
                    baseline = None
                    for name, tokens in variants.items():
                        with torch.no_grad(), codec.autocast():
                            rgb, _ = model.decoder(tokens, reference)
                        rgb = rgb.float()
                        baseline = rgb if baseline is None else baseline
                        row = {**info, "variant": name, "diagnostic_only": True,
                               **metrics(frame, rgb), **detail_metrics(rgb, frame),
                               "output_change_mse": float(F.mse_loss(rgb, baseline))}
                        bypass.append(row)
                        append_row(output / "bypass.jsonl", row)
                        if frame_index == 0 and split == "validation":
                            save_image(output / "samples" / f"{source['content_id'].replace(':', '_')}_{name}.png", rgb)
                    gradient_rows.append({**info, **gradients(model, z, discrete, reference, frame, 2048)})
                with torch.no_grad():
                    packet, state = codec.encode_frame(frame, state, 2048, source["content_id"])
                print(json.dumps({"E0": source["content_id"], "frame": frame_index}), flush=True) if frame_index % 16 == 0 else None
    summary = {"environment": environment(), "checkpoint_sha256": file_hash(args.checkpoint),
               "completed_steps": checkpoint["completed_steps"], "manifest_identity": index["identity"],
               "inputs": inputs, "diagnostic_only": True, "history": "real K2048 packet local decoder only",
               "representations": summarize_representations(representations), "bypass": summarize_bypass(bypass)}
    write_json(output / "summary.json", summary)
    write_json(output / "gradients.json", gradient_rows)
    write_json(output / "metric_manifest.json", metric_manifest())
    write_report(output, summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="checkpoints/adaptok_hr/joint/step_001000.pt")
    parser.add_argument("--index", default="data/adaptok_hr/index.json")
    parser.add_argument("--frames", type=int, default=64)
    parser.add_argument("--output", default="results/adaptok_hr_fine_recovery/E0")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
