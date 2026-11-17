"""Initialize the 8192-code vocabulary exclusively from Train Split pre-VQ features."""

import argparse
import copy

import torch

from cp_adaptok.runtime import resolve, seed_everything, write_json
from experiments.fixed2048.checkpoint import load_reconstructor, save_training
from experiments.fixed2048.cli import read_config
from experiments.fixed2048.data import read_clip
from experiments.fixed2048.train import require_training_config, training_sources


@torch.no_grad()
def collect_train_features(model, config, videos=4, frames=8):
    sources, hashes = training_sources(config)
    samples, used = [], []
    for source in sources[:videos]:
        clip = read_clip(source, frames, crop=(1080, 1920))
        state = model.empty_state()
        for index, frame in enumerate(clip):
            encoded = model.encode(frame[None].to(model.device), state, index)
            features = model.tokens(encoded)
            samples.append(model.vq.project_features(features).detach().cpu())
            output = model.reconstruct(encoded, "R1")
            state = model.advance(encoded, output, "full2048")
        used.append({"video": source["content_id"], "start": 0, "frames": frames, "split": "train"})
    return torch.cat(samples).flatten(0, 1), {
        "samples": used,
        "manifest_hashes": hashes,
        "sample_count": sum(s.shape[0] * s.shape[1] for s in samples),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/fixed2048.yaml")
    parser.add_argument("--source-checkpoint", required=True)
    parser.add_argument("--output", default="checkpoints/fixed2048/vq/calibrated.pt")
    parser.add_argument("--report", default="results/fixed2048/quantized/calibration.json")
    args = parser.parse_args()
    output = resolve(args.output)
    if output.exists():
        raise FileExistsError("calibration checkpoint already exists")
    config = copy.deepcopy(read_config(args.config))
    require_training_config(config)
    seed_everything(config["seed"], config["device"], 0.9)
    model, identity = load_reconstructor(args.source_checkpoint, config["device"])
    features, provenance = collect_train_features(model, config)
    model.vq.initialize_codebook(features, config["seed"], iterations=20)
    config["source_checkpoint"] = args.source_checkpoint
    config["training"]["path"] = "R2"
    model.configure_training("R2")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    generator = torch.Generator().manual_seed(config["seed"])
    provenance.update(
        method="existing normalized kmeans_centers",
        iterations=20,
        seed=config["seed"],
        continuous_source=identity,
        validation_test_features_used=False,
    )
    save_training(
        output,
        model,
        optimizer,
        scheduler,
        generator,
        0,
        config,
        {"calibration": provenance, "train": [], "validation": []},
        identity,
    )
    write_json(args.report, provenance)


if __name__ == "__main__":
    main()
