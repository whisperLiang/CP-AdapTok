"""Post-update probes: sampled training assignments and deterministic assignments, without updates."""

import argparse
import json
from pathlib import Path

import torch

from cp_adaptok.data import read_video, sample_training_spec
from cp_adaptok.diagnostics import vq_diagnostics
from cp_adaptok.runtime import build_model, load_checkpoint, seed_everything, write_json
from cp_adaptok.training import resolution_at, sequence_length, set_quantization_step, training_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoints", nargs="+")
    args = parser.parse_args()
    for checkpoint in args.checkpoints:
        artifact = load_checkpoint(checkpoint)
        config = artifact["config"]
        seed_everything(config["seed"], config["device"], 0.8)
        model, _, _ = build_model(config, config["device"], checkpoint_override=checkpoint)
        model.train()
        model.tokenizer.diagnostics_enabled = True
        set_quantization_step(model, config, artifact["step"])
        generator = torch.Generator().manual_seed(config["training"].get("fixed_sample_seed", 19))
        length = sequence_length(config["training"], artifact["step"])
        spec = sample_training_spec(config["data"]["train"][0], length, generator)
        clip = read_video(spec, resolution_at(config["training"], artifact["step"]), generator)
        state, records = model.empty_state(), []
        with torch.no_grad():
            for frame in clip:
                encoded = model.encode_full(frame[None].to(model.device), state)
                records.append(
                    {
                        "frame_type": encoded.context.frame_type,
                        "query_layers": getattr(model.tokenizer, "last_diagnostics", {}).copy(),
                        **vq_diagnostics(model.vq, encoded.features, encoded.vq),
                    }
                )
                _, feature = model.decode_embeddings(
                    encoded.vq.embeddings[:, : model.config.k_ref], encoded.context
                )
                state = encoded.context.commit(feature, training_digest(encoded, model.config.k_ref)).detach()
        output = Path(checkpoint).parent / "post_update_quantizer.json"
        write_json(
            output,
            {
                "checkpoint": checkpoint,
                "completed_updates": artifact["step"],
                "optimizer_updated": False,
                "policy": model.vq.strategy,
                "temperature": model.vq.temperature,
                "records": records,
            },
        )
        print(json.dumps({"checkpoint": checkpoint, "step": artifact["step"]}), flush=True)
        del model, artifact
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
