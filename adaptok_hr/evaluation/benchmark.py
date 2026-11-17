"""Synchronized full codec and component benchmarks, without resident quality networks."""

import argparse
import hashlib
import json

import torch

from cp_adaptok.metrics import summarize_latency

from ..codec.bitstream import HEADER, Packet, pack_codes, parse, serialize, unpack_codes
from ..codec.codec import Codec
from ..data.dataset import read_clip
from ..data.manifest import load_index
from ..models.temporal_state import TemporalState, frame_context
from ..runtime import environment, file_hash, seed_everything, write_json
from ..training.checkpoint import load_model
from .evaluate import timed


class ComponentEvents:
    def __init__(self, groups):
        self.rows, self.handles = {}, []
        for name, modules in groups.items():
            self.rows[name] = []
            for module in modules:
                self.handles.append(module.register_forward_pre_hook(self.start(name)))
                self.handles.append(module.register_forward_hook(self.stop(name)))

    def start(self, name):
        def hook(module, args):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.rows[name].append([event, None])
        return hook

    def stop(self, name):
        def hook(module, args, output):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            self.rows[name][-1][1] = event
        return hook

    def finish(self):
        torch.cuda.synchronize()
        for handle in self.handles:
            handle.remove()
        return {key: sum(a.elapsed_time(b) for a, b in events) for key, events in self.rows.items()}


@torch.inference_mode()
def component_sample(codec, frame, state, budget):
    model, device = codec.model, codec.device
    _, _, reference = frame_context(state, model.config.gop)
    events = ComponentEvents({"conv_encoder": list(model.encoder.stages),
                              "encoder_window_attention": list(model.encoder.attention.values())})
    with codec.autocast():
        features, encoder_wall = timed(lambda: model.encoder(frame, reference), device)
        encoder_parts = events.finish()
        values, generate = timed(lambda: model.tokenizer(features), device)
        encoded, quantize = timed(lambda: model.vq(values), device)
        embeddings, lookup = timed(lambda: model.vq.lookup(encoded.ids[:, :budget]), device)
        maps, scatter = timed(lambda: model.positions.scatter(embeddings), device)
        decoder_events = ComponentEvents({"decoder_window_attention": list(model.decoder.attention)})
        low, fusion = timed(lambda: model.decoder.fuse_maps(maps, reference), device)
        decoder_parts = decoder_events.finish()
        rgb, upscale = timed(lambda: model.decoder.upsample_rgb(low, reference), device)
    ids, ids_cpu = timed(lambda: encoded.ids[0, :budget].cpu().tolist(), device)
    payload, packing = timed(lambda: pack_codes(ids), torch.device("cpu"))
    _, unpacking = timed(lambda: unpack_codes(payload, budget), torch.device("cpu"))
    index, kind, _ = frame_context(state, model.config.gop)
    header = Packet(index, kind == "I", model.config.gop, model.config.height, model.config.width,
                    tuple(ids), b"x" * 16, codec.codebook_identity, codec.model_identity,
                    bytes(32) if kind == "I" else state.chain)
    packet, framing = timed(lambda: serialize(header), torch.device("cpu"))
    _, parsing = timed(lambda: parse(packet), torch.device("cpu"))
    _, update = timed(lambda: TemporalState(b"x" * 16, index, kind, model.config.gop, rgb, low,
                                            chain=hashlib.sha256(packet).digest()), device)
    return {**encoder_parts, **decoder_parts,
            "all_window_attention_cuda": encoder_parts["encoder_window_attention"] + decoder_parts["decoder_window_attention"],
            "encoder_total_wall": encoder_wall, "token_generation": generate, "vq": quantize,
            "token_lookup": lookup, "token_scatter": scatter, "decoder_low_fusion_attention": fusion,
            "decoder_upsampling": upscale, "state_update": update, "ids_gpu_to_cpu": ids_cpu,
            "bit_packing_cpu": packing, "bit_unpacking_cpu": unpacking,
            "framing_including_pack_crc": framing, "parsing_including_unpack_crc": parsing}


@torch.inference_mode()
def flop_analysis(model, frame, state):
    from torch.utils.flop_counter import FlopCounterMode

    results = {}
    for name, reference in (("I", None), ("P", state)):
        with FlopCounterMode(display=False) as counter:
            model.forward_frame(frame, reference, model.config.budgets[-1])
        results[name] = {"flops": counter.get_total_flops(),
                         "gflops": counter.get_total_flops() / 1e9,
                         "operators": {str(k): int(v) for k, v in counter.flop_counts["Global"].items()}}
    return {"parameters": sum(p.numel() for p in model.parameters()), "frames": results,
            "convention": "PyTorch FlopCounterMode; multiply-add=2; counts supported conv/mm/SDPA operators; excludes interpolation,norm,activation,packing and CPU operations"}


def condition_summary(samples, parts):
    return {"encoder": summarize_latency([r["encoder_ms"] for r in samples]),
                         "sender_with_local_decode": summarize_latency([r["sender_ms"] for r in samples]),
                         "decoder": summarize_latency([r["decoder_ms"] for r in samples]),
                         "joint_sender_plus_receiver": summarize_latency([r["joint_ms"] for r in samples]),
                         "components_ms": {k: summarize_latency([r[k] for r in parts]) for k in parts[0]},
                         "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                         "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
            "samples": samples, "component_samples": parts}


def benchmark_condition(codec, frame, state, budget, stream, warmup, repeats):
    for _ in range(warmup):
        packet, _ = codec.encode_frame(frame, state, budget, stream)
        codec.decode_frame(packet, state)
    torch.cuda.reset_peak_memory_stats()
    samples, parts = [], []
    for _ in range(repeats):
        with codec.autocast():
            _, encoder_ms = timed(lambda: codec.model.token_ids(frame, state, budget), codec.device)
        (packet, _), sending_ms = timed(lambda: codec.encode_frame(frame, state, budget, stream), codec.device)
        _, decoder_ms = timed(lambda: codec.decode_frame(packet, state), codec.device)
        samples.append({"encoder_ms": encoder_ms, "sender_ms": sending_ms, "decoder_ms": decoder_ms,
                        "joint_ms": sending_ms + decoder_ms, "bytes": len(packet)})
        parts.append(component_sample(codec, frame, state, budget))
    return condition_summary(samples, parts)


@torch.inference_mode()
def benchmark(model, source, budget, warmup=5, repeats=20):
    device = next(model.parameters()).device
    if device.type != "cuda" or warmup < 1 or repeats < 2:
        raise ValueError("benchmark requires CUDA, warmup>=1, repeats>=2")
    clip, metadata = read_clip(source, 0, 2, model.config.height, model.config.width)
    first, second = clip[0:1].to(device), clip[1:2].to(device)
    codec = Codec(model.eval(), amp_bf16=True)
    _, history = codec.encode_frame(first, None, budget, source["content_id"])
    results = {kind: benchmark_condition(codec, frame, state, budget, source["content_id"], warmup, repeats)
               for kind, frame, state in (("I", first, None), ("P", second, history))}
    return {"budget": budget, "warmup": warmup, "repeats": repeats, "batch_size": 1,
            "input": metadata, "results": results,
            "scope": "resident pixels, BF16 NN with FP32 VQ; synchronized wall; encoder excludes packing/local decode; decoder includes parse/lookup/state; joint includes encode+sender local decode+independent receiver decode+packing; excludes load,H2D,disk,metrics",
            "component_scope": "separate component probes, not summed for throughput; encoder CUDA events split conv/windows; total encoder and downstream values synchronized wall; CRC framing/parsing include pack/unpack",
            "flops": flop_analysis(model, first, history), "header_bytes": HEADER.size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--index", default="data/adaptok_hr/index.json")
    parser.add_argument("--content-id", default="DAVIS2017:car-roundabout")
    parser.add_argument("--budgets", nargs="*", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    seed_everything(20261117)
    torch.set_num_threads(4)
    model, _ = load_model(args.checkpoint)
    index = load_index(args.index)
    source = next(r for r in index["splits"]["validation"] if r["content_id"] == args.content_id)
    results = {str(k): benchmark(model, source, k, args.warmup, args.repeats)
               for k in (args.budgets or model.config.budgets)}
    write_json(args.output, {"environment": environment(), "checkpoint_sha256": file_hash(args.checkpoint),
                             "budgets": results})
    print(json.dumps({k: v["results"]["P"]["joint_sender_plus_receiver"] for k, v in results.items()}))


if __name__ == "__main__":
    main()
