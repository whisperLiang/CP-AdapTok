import argparse
import json
import math
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict

import torch

from cp_adaptok.cli.common import add_inference_options, inference_config, load_inference
from cp_adaptok.data import read_video, save_rgb
from cp_adaptok.metrics import MetricSuite
from cp_adaptok.runtime import (
    MissingPrerequisites,
    metadata,
    report_missing,
    resolve,
    write_json,
)
from cp_adaptok.streaming.sender import CodecEncoder
from cp_adaptok.streaming.session import AllocationConfig, SocketReceiver, SocketSender


def configure_connection(connection):
    connection.settimeout(30)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)


@torch.no_grad()
def receive_frames(listener, model, output, save_frames, ack_delay_s):
    frames = []

    def display(t, k, reconstruction):
        if save_frames:
            save_rgb(output / f"frame{t:03d}_k{k}.png", reconstruction)
        frames.append((t, k, reconstruction.cpu()))

    with listener.accept()[0] as connection:
        configure_connection(connection)
        report = SocketReceiver(connection, model, display, ack_delay_s).run()
    return report, frames


def send_video(connection, encoder, clip, budgets, allocation, rates, fps):
    sender = SocketSender(connection, encoder, budgets, allocation)
    origin, rows = time.perf_counter(), []
    for t, rgb in enumerate(clip):
        # Live-source clock: slow base delivery creates real backlog for later frames.
        release = time.perf_counter() if allocation is None else origin + t / fps
        while (remaining := release - time.perf_counter()) > 0:
            time.sleep(min(1, remaining))
        rows.append(sender.send_frame(rgb[None].to(encoder.model.device), rates[t], release))
    sender.finish()
    return sender, rows


def observed_rates(bandwidth_kbps, count, allocation):
    if bandwidth_kbps is None:
        if allocation is not None:
            raise ValueError("online streaming requires --bandwidth-kbps")
        return [None] * count
    if allocation is None:
        raise ValueError("bandwidth pacing requires an online allocation policy")
    rates = tuple(bandwidth_kbps)
    if not rates or any(not math.isfinite(r) or r <= 0 for r in rates):
        raise ValueError("bandwidth observations must be finite and positive")
    return [rates[min(t, len(rates) - 1)] * 1000 for t in range(count)]


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


def verify_delivery(receiver, sender, rows):
    if receiver["received_bytes"] != sender.writer.bytes_sent or receiver["ack_bytes"] != sender.ack_bytes:
        raise RuntimeError("socket byte accounting mismatch")
    by_frame = {row["frame"]: row for row in rows}
    for event in receiver["events"]:
        row = by_frame[event["frame"]]
        if event["state_fingerprint"] != row["state_fingerprint"]:
            raise RuntimeError("socket receiver state drift")
        event["elapsed_since_release_s"] = event["display_time"] - row["release_time"]
        if "deadline" in row:
            event["deadline_lateness_s"] = max(0, event["display_time"] - row["deadline"])
        if event["tokens"] == row["delivered_tokens"] and event["packet_sha256"] != row["packet_sha256"]:
            raise RuntimeError("socket prefix bytes changed")


def delivery_report(receiver, sender, rows, rates, elapsed, save_frames):
    codec_bytes = sum(row["codec_and_framing_bytes"] for row in rows)
    return {
        "events": receiver["events"],
        "sender_frames": rows,
        "sender_base_commits": sender.encoder.base_commits,
        "receiver_base_commits": receiver["base_commits"],
        "codec_and_framing_bytes": codec_bytes,
        "forward_control_and_framing_bytes": sender.writer.bytes_sent - codec_bytes,
        "ack_control_and_framing_bytes": receiver["ack_bytes"],
        "total_application_bytes": sender.writer.bytes_sent + receiver["ack_bytes"],
        "elapsed_s": elapsed,
        "fps_with_transport_and_display": len(rows) / elapsed,
        "fps_with_png_display_and_acks": len(rows) / elapsed if save_frames else None,
        "saved_png_frames": save_frames,
        "encoding_ms": [row["encoding_ms"] for row in rows],
        "bandwidth_kbps_observations": [r / 1000 if r is not None else None for r in rates],
        "pacing": "forward application records serialized at current rate; ACK path unpaced",
        "queue": "one frame in flight; no application send queue; source backlog retained",
        "omitted_enhancement_packets": sum(row["omitted_enhancement_packets"] for row in rows),
        "base_wait_frames": sum(row.get("base_wait_required", False) for row in rows),
        "deadline_missed_displays": sum(e.get("deadline_lateness_s", 0) > 0 for e in receiver["events"]),
    }


def source_spec(config, max_frames):
    spec = dict(config["data"]["test"][0])
    if max_frames is not None:
        if max_frames < 1:
            raise ValueError("max_frames must be positive")
        spec["frames"] = min(max_frames, spec["frames"])
    fps = float(spec["fps"])
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError("invalid source FPS")
    return spec, fps


def display_budgets(c, config, progressive, allocation):
    budgets = sorted(set([c.k_ref, *config["evaluation"]["budgets"], c.n_max]))
    if allocation is not None and allocation.fixed_k is not None:
        budgets = sorted(set([*budgets, allocation.fixed_k]))
    if allocation is None and not progressive:
        budgets = [c.n_max]
    return budgets


@torch.no_grad()
def localhost_demo(
    model,
    config,
    progressive,
    output,
    max_frames=None,
    *,
    allocation=None,
    bandwidth_kbps=None,
    save_frames=True,
    ack_delay_s=0,
):
    output = resolve(output)
    output.mkdir(parents=True, exist_ok=True)
    spec, fps = source_spec(config, max_frames)
    clip = read_video(spec)
    rates = observed_rates(bandwidth_kbps, len(clip), allocation)
    budgets = display_budgets(model.config, config, progressive, allocation)
    encoder = CodecEncoder(model)
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.settimeout(10)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        with ThreadPoolExecutor(1) as pool:
            future = pool.submit(receive_frames, listener, model, output, save_frames, ack_delay_s)
            start = time.perf_counter()
            with socket.create_connection(listener.getsockname(), timeout=30) as connection:
                configure_connection(connection)
                sender, rows = send_video(connection, encoder, clip, budgets, allocation, rates, fps)
            receiver, reconstructions = future.result(timeout=30)
            elapsed = time.perf_counter() - start
    qualities, improvements = progressive_quality(clip, reconstructions)
    verify_delivery(receiver, sender, rows)
    return {
        **delivery_report(receiver, sender, rows, rates, elapsed, save_frames),
        "session_protocol_version": 1,
        "mode": "localhost_tcp",
        "progressive": progressive,
        "source": spec,
        "sample_count": len(clip),
        "allocation": asdict(allocation) if allocation is not None else None,
        "quality": qualities,
        "base_to_final": improvements,
        "real_entropy_bitstream": True,
        "link_layer_overhead": "TCP/IP/link packet headers not captured",
        "receiver_inputs": "model weights and socket records only",
    }


def allocation_options(parser):
    parser.add_argument("--allocation", choices=("fixed", "rate-only", "scorer"))
    parser.add_argument("--fixed-k", type=int)
    parser.add_argument(
        "--bandwidth-kbps", type=float, nargs="+", help="current per-frame rates; hold last value"
    )
    parser.add_argument("--playback-delay-ms", type=float, default=250)
    parser.add_argument("--decode-reserve-ms", type=float, default=25)
    parser.add_argument("--buffer-ms", type=float, default=0)
    parser.add_argument("--enhancement-cap", type=int, help="truncate unsent enhancements at this prefix")
    parser.add_argument(
        "--ack-delay-ms", type=float, default=0, help="receiver delay before each display ACK"
    )
    parser.add_argument(
        "--no-save-frames", action="store_true", help="display is tensor ready, without PNG I/O"
    )


def parse_allocation(parser, args):
    if args.allocation is None:
        if args.bandwidth_kbps is not None or args.fixed_k is not None or args.enhancement_cap is not None:
            parser.error("bandwidth/fixed-k/enhancement-cap require --allocation")
        return None
    try:
        policy = AllocationConfig(
            mode=args.allocation,
            playback_delay_s=args.playback_delay_ms / 1000,
            decode_reserve_s=args.decode_reserve_ms / 1000,
            buffer_s=args.buffer_ms / 1000,
            fixed_k=args.fixed_k,
            enhancement_cap=args.enhancement_cap,
        )
        observed_rates(args.bandwidth_kbps, 1, policy)
    except ValueError as error:
        parser.error(str(error))
    return policy


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Append-only rANS over real loopback TCP; base display ACK gates enhancements"
    )
    add_inference_options(parser, "results/stream_demo")
    parser.add_argument("--mode", choices=("localhost",), required=True)
    parser.add_argument("--progressive", action="store_true")
    parser.add_argument("--max-frames", type=int)
    allocation_options(parser)
    args = parser.parse_args(argv)
    allocation = parse_allocation(parser, args)
    config = inference_config(args)
    try:
        model, status = load_inference(config, args, "stream_demo")
        report = metadata(config, status)
        report["artifact"] = model.artifact_provenance
        report.update(
            localhost_demo(
                model.eval(),
                config,
                args.progressive,
                args.output,
                args.max_frames,
                allocation=allocation,
                bandwidth_kbps=args.bandwidth_kbps,
                save_frames=not args.no_save_frames,
                ack_delay_s=args.ack_delay_ms / 1000,
            )
        )
        write_json(resolve(args.output) / "stream_demo.json", report)
        print(
            json.dumps(
                {
                    "report": str(resolve(args.output) / "stream_demo.json"),
                    "status": status,
                    "bytes": report["total_application_bytes"],
                    "quality": report["base_to_final"],
                    "base_wait_frames": report["base_wait_frames"],
                    "omitted_enhancement_packets": report["omitted_enhancement_packets"],
                }
            )
        )
    except MissingPrerequisites as error:
        report_missing(resolve(args.output) / "prerequisites.json", error, config)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
