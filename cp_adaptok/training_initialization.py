"""Collect actual continuous warmup features before one-time codebook initialization."""

import torch

from cp_adaptok.data import read_video, sample_training_spec
from cp_adaptok.runtime import write_json


@torch.no_grad()
def initialize_from_training_data(model, config, output):
    settings = config["training"]["vq"]["initialization"]
    generator = torch.Generator().manual_seed(config["seed"] + 71)
    collected, count = [], 0
    policy = model.vq.strategy, model.vq.temperature, model.vq.sampling_topk
    model.vq.set_training_policy("continuous")
    sources = config["data"]["train"]
    try:
        for i in range(settings.get("clips", 16)):
            source = sources[i % len(sources)]
            spec = sample_training_spec(source, min(4, source["frames"]), generator)
            clip = read_video(spec, config["training"]["crop"], generator)
            state = model.empty_state()
            for frame in clip:
                encoded = model.encode_full(frame[None].to(model.device), state)
                z = model.vq.project_features(encoded.features).flatten(0, 1).cpu()
                collected.append(z)
                count += len(z)
                _, feature = model.decode_embeddings(
                    encoded.vq.embeddings[:, : model.config.k_ref], encoded.context
                )
                state = encoded.context.commit(feature, "continuous-initialization").detach()
            if count >= settings.get("max_samples", 32768):
                break
        samples = torch.cat(collected)[: settings.get("max_samples", 32768)]
        model.vq.initialize_codebook(samples, config["seed"], settings.get("iterations", 20))
        torch.save(samples, output / "codebook_initialization_samples.pt")
        write_json(
            output / "codebook_initialization.json",
            {
                "samples": len(samples),
                "distinct_samples": len(samples.unique(dim=0)),
                "codebook_size": model.config.codebook_size,
                "seed": config["seed"],
                "source_split": "train",
                "source_paths": [x["path"] for x in sources],
                "method": "reproducible spherical kmeans",
            },
        )
    finally:
        model.vq.set_training_policy(*policy)
