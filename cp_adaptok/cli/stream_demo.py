import argparse
import json
import socket
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from cp_adaptok.data import read_video, save_rgb
from cp_adaptok.metrics import MetricSuite
from cp_adaptok.runtime import (
    MissingPrerequisites,
    build_model,
    metadata,
    preflight,
    read_config,
    report_missing,
    resolve,
    seed_everything,
    write_json,
)
from cp_adaptok.streaming.receiver import CodecDecoder
from cp_adaptok.streaming.sender import CodecEncoder
from cp_adaptok.streaming.transport import receive_record, send_record


@torch.no_grad()
def receive_frames(listener, model, frame_count, budgets, output):
    decoder = CodecDecoder(model)
    events, frames, ack_bytes = [], [], 0
    with listener.accept()[0] as connection:
        connection.settimeout(30)
        connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        for t in range(frame_count):
            decoder.start_frame(receive_record(connection))
            received = 0
            for k in budgets:
                while received < k:
                    decoder.append_packets([receive_record(connection)])
                    received += model.config.packet_tokens
                reconstruction = decoder.decode_prefix(k, use_cache=True)
                # The display artifact exists before we acknowledge and allow enhancements.
                save_rgb(output / f"frame{t:03d}_k{k}.png", reconstruction)
                frames.append((t, k, reconstruction.cpu()))
                events.append(
                    {
                        "frame": t,
                        "tokens": k,
                        "display_time": time.perf_counter(),
                        "state_fingerprint": decoder.get_state().fingerprint(),
                        "reconstructions": decoder.reconstructions,
                        **decoder.last_timing_ms,
                    }
                )
                ack_bytes += send_record(connection, f"DISPLAYED:{t}:{k}".encode())
    return events, frames, ack_bytes


def send_video(connection, encoder, clip, budgets):
    byte_count, fingerprints, encoding = 0, [], []
    for t, rgb in enumerate(clip):
        encoded = encoder.encode_frame(rgb[None].to(encoder.model.device), budgets=budgets)
        encoding.append(encoded.timing_ms)
        fingerprints.append(encoder.get_state().fingerprint())
        byte_count += send_record(connection, encoded.header_bytes)
        count = 0
        for k in budgets:
            end = k // encoder.model.config.packet_tokens
            for packet in encoded.packets[count:end]:
                byte_count += send_record(connection, packet)
            count = end
            if receive_record(connection).decode() != f"DISPLAYED:{t}:{k}":
                raise RuntimeError("receiver did not acknowledge the displayed prefix")
    return byte_count, fingerprints, encoding


def progressive_quality(clip, reconstructions):
    suite = MetricSuite("cpu", ("psnr",))
    qualities = [
        {"frame": t, "tokens": k, **suite(clip[t : t + 1], decoded)} for t, k, decoded in reconstructions
    ]
    improvements = []
    for t in range(len(clip)):
        rows = [q for q in qualities if q["frame"] == t]
        improvements.append(
            {
                "frame": t,
                "base_mse": rows[0]["mse"],
                "final_mse": rows[-1]["mse"],
                "quality_improved": rows[-1]["mse"] < rows[0]["mse"],
            }
        )
    return qualities, improvements


@torch.no_grad()
def localhost_demo(model, config, progressive, output, max_frames=None):
    output = resolve(output)
    output.mkdir(parents=True, exist_ok=True)
    spec = dict(config["data"]["test"][0])
    if max_frames:
        spec["frames"] = min(max_frames, spec["frames"])
    clip = read_video(spec)
    c = model.config
    budgets = sorted(set([c.k_ref, *config["evaluation"]["budgets"], c.n_max])) if progressive else [c.n_max]
    if any(k < c.k_ref or k > c.n_max or k % c.packet_tokens for k in budgets):
        raise ValueError("invalid display packet boundary")
    encoder = CodecEncoder(model)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(receive_frames, listener, model, len(clip), budgets, output)
            start = time.perf_counter()
            with socket.create_connection(listener.getsockname(), timeout=30) as connection:
                connection.settimeout(30)
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                byte_count, fingerprints, encoding = send_video(connection, encoder, clip, budgets)
            events, reconstructions, ack_bytes = future.result(timeout=30)
            elapsed = time.perf_counter() - start
    qualities, improvements = progressive_quality(clip, reconstructions)
    for event in events:
        if event["state_fingerprint"] != fingerprints[event["frame"]]:
            raise RuntimeError("socket receiver state drift")
    return {
        "mode": "localhost_tcp",
        "progressive": progressive,
        "source": spec,
        "sample_count": len(clip),
        "events": events,
        "quality": qualities,
        "base_to_full": improvements,
        "codec_and_framing_bytes": byte_count,
        "ack_control_and_framing_bytes": ack_bytes,
        "total_application_bytes": byte_count + ack_bytes,
        "elapsed_s": elapsed,
        "fps_with_png_display_and_acks": len(clip) / elapsed,
        "encoding_ms": encoding,
        "real_entropy_bitstream": True,
        "link_layer_overhead": "TCP/IP/link packet headers not captured",
        "receiver_inputs": "model weights and socket records only",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Append-only rANS over real loopback TCP; base display ACK gates enhancements"
    )
    parser.add_argument("--config", default="configs/s1_tokenizer.yaml")
    parser.add_argument("--mode", choices=("localhost",), required=True)
    parser.add_argument("--progressive", action="store_true")
    parser.add_argument("--allow-untrained", action="store_true")
    parser.add_argument("--checkpoint")
    parser.add_argument("--device")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--output", default="results/stream_demo")
    args = parser.parse_args(argv)
    config = read_config(args.config)
    if args.checkpoint:
        config["paths"]["checkpoint"] = args.checkpoint
    device = args.device or config["device"]
    try:
        preflight(config, "stream_demo", args.allow_untrained)
        seed_everything(config["seed"], device, config["gpu_memory_fraction"])
        model, status, _ = build_model(config, device, args.allow_untrained, args.checkpoint)
        report = metadata(config, status)
        report["artifact"] = model.artifact_provenance
        report.update(localhost_demo(model.eval(), config, args.progressive, args.output, args.max_frames))
        write_json(resolve(args.output) / "stream_demo.json", report)
        print(
            json.dumps(
                {
                    "report": str(resolve(args.output) / "stream_demo.json"),
                    "status": status,
                    "bytes": report["total_application_bytes"],
                    "quality": report["base_to_full"],
                }
            )
        )
    except MissingPrerequisites as error:
        report_missing(resolve(args.output) / "prerequisites.json", error, config)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
