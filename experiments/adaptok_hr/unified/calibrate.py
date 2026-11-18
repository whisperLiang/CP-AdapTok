"""Freeze independent Alex/VGG weights from Train-only RGB gradient measurements."""

import argparse
import json
from statistics import median

import torch

from adaptok_hr.data.dataset import read_clip
from adaptok_hr.data.manifest import load_index
from adaptok_hr.evaluation.evaluate import metric_manifest
from adaptok_hr.runtime import environment, file_hash, seed_everything, write_json
from adaptok_hr.training.checkpoint import load_model
from adaptok_hr.training.losses import ReconstructionLoss

from .run import DEFAULT_PARENTS, ROOT, parent_path


def calibrate(parent_root):
    output = ROOT / "calibration.json"
    if output.exists():
        return
    seed_everything(20261201)
    torch.set_num_threads(4)
    index = load_index("data/adaptok_hr/index.json")
    source = index["splits"]["train"][0]
    clip, metadata = read_clip(source, 0, 2)
    rows, ratios = [], []
    networks = {net: ReconstructionLoss({"l1": 1., "perceptual": 1.}, "cuda",
                 {"mode": "unified", "network": net, "crop_size": 256,
                  "global_size": [270, 480], "crop_seed": 20261201}) for net in ("alex", "vgg")}
    for seed in (1, 2):
        path = parent_path(parent_root, seed)
        model, _ = load_model(path)
        networks["alex"].crop_rng.manual_seed(20261201 + (seed - 1) * 10)
        state = None
        for index, cpu in enumerate(clip):
            frame = cpu[None].cuda()
            views = networks["alex"].sample_views(frame)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                encoded = model.encode_features(frame, state)
                outputs = {k: model.forward_frame(frame, state, k, encoded) for k in model.config.budgets}
            for budget, decoded in outputs.items():
                row = {"seed": seed, "frame": index, "budget": budget,
                       "parent_checkpoint_sha256": file_hash(path), "crops": views, "networks": {}}
                for net, losses in networks.items():
                    rgb = decoded.rgb.detach().float().requires_grad_()
                    _, terms = losses.distortion(rgb, frame, views)
                    gradient, = torch.autograd.grad(terms["perceptual"], rgb)
                    row["networks"][net] = {**{k: float(v.detach()) for k, v in terms.items()},
                                             "rgb_perceptual_gradient_l2": float(gradient.norm())}
                ratios.append(row["networks"]["alex"]["rgb_perceptual_gradient_l2"] /
                              row["networks"]["vgg"]["rgb_perceptual_gradient_l2"])
                rows.append(row)
            state = outputs[2048].state.detach()
        del outputs, encoded, model
    weights = {"alex": .1, "vgg": round(.1 * median(ratios), 8)}
    write_json(output, {"environment": environment(), "input": metadata, "split": "train",
                        "method": "VGG lambda = Alex lambda * median(Alex/VGG RGB perceptual-gradient L2), equal-view mean; not an assertion of network-scale equivalence",
                        "chosen_weights": weights, "gradient_ratios": ratios, "rows": rows,
                        "metrics": metric_manifest()})
    print(json.dumps(weights), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-root", type=type(DEFAULT_PARENTS), default=DEFAULT_PARENTS)
    calibrate(parser.parse_args().parent_root)
