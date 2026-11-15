"""Numerical order diagnostic on retained trained-model packets, never a new stream."""

import argparse
import copy

import torch

from cp_adaptok.runtime import build_model, read_config, resolve, seed_everything, write_json
from cp_adaptok.streaming.receiver import CodecDecoder
from scripts.verify_historical_stream import wire_records


@torch.no_grad()
def grids_in_orders(projector, embeddings):
    query = projector.new_cache(1).query
    deltas = [projector._group_delta(query, embeddings[:, first:first + projector.packet_tokens], first)
              for first in range(0, embeddings.shape[1], projector.packet_tokens)]
    grids = []
    for values in (deltas, list(reversed(deltas))):
        total = torch.zeros_like(query)
        for value in values:
            total = total + value
        grids.append(projector._finish(total))
    return grids


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="results/prefix_utilization/full_1080p/C2/config.json")
    parser.add_argument("--checkpoint", default="results/prefix_utilization/full_1080p/C2/calibrated.pt")
    parser.add_argument("--packet-file", default="results/prefix_utilization/full_1080p/C2/basketball/frame000.cpat")
    parser.add_argument("--output", default="results/prefix_utilization/performance/accumulation_order.json")
    args = parser.parse_args()
    config = read_config(args.config)
    seed_everything(config["seed"], config["device"], config["gpu_memory_fraction"])
    model, _, _ = build_model(config, config["device"], checkpoint_override=args.checkpoint)
    model.eval()
    receiver = CodecDecoder(model)
    records = wire_records(resolve(args.packet_file))
    receiver.start_frame(records[0])
    receiver.append_packets(records[1:])
    ids = torch.tensor([receiver.stream.token_ids], device=model.device)
    embeddings = model.vq.embed(ids)
    results = {}
    for precision in ("float64", "float32"):
        diagnostic = copy.deepcopy(model.decoder.projector)
        diagnostic.accumulation_dtype = getattr(torch, precision)
        grids = grids_in_orders(diagnostic, embeddings)
        images = [model.decode_latent(model.decoder.from_grid(grid, receiver.context.latent_shape,
                    receiver.context.frame_type), receiver.context)[0] for grid in grids]
        results[precision] = {"grid_reverse_max_error": (grids[0] - grids[1]).abs().max().item(),
                              "image_reverse_max_error": (images[0] - images[1]).abs().max().item()}
    write_json(args.output, {"source": args.packet_file, "model_id": model.model_id,
               "orders": ["ascending packet order", "reversed summation only (not packet reordering)"],
               "results": results, "production_order": "ascending packets for both full and cached paths",
               "production_precision": model.config.attention_precision, "diagnostic_only": True})


if __name__ == "__main__":
    main()
