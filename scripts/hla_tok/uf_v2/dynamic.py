"""Independent Base, changing bandwidth, and loss recovery on real continuous clips."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from time import perf_counter

import torch

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.entropy import EntropyTables
from hla_tok.uf.official import OfficialAssets, sha256_file
from hla_tok.uf.wire import WireError
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf_v2.evaluate import allowance, decoder_for, encode, load_codec
from scripts.hla_tok.v2.protocol import setup

CYCLE = (5, 40, 10, 20, 5)


def native_z_tag(endpoint):
    digest = hashlib.sha256()
    for net in (endpoint.intra, endpoint._video()):
        for table in net.bit_estimator_z.get_cdf_info():
            digest.update(table.tobytes())
    return digest.hexdigest()


def probability_evidence(codec, bases, table):
    native = native_z_tag(codec.native)
    if any(native_z_tag(receiver) != native for receiver in bases):
        raise ValueError("independent encoder and decoder rebuilt different native z CDF tables")
    if table is None:
        enhanced = EntropyTables.from_scales(tuple(codec.model.scales().tolist()))
        enhancement = hashlib.sha256(repr(enhanced.cdf).encode()).hexdigest()
    else:
        enhancement = sha256_file(table)
    return {"encoder_and_decoder_native_z_sha256": native, "enhancement_probability_sha256": enhancement}


def snapshot(endpoint):
    with torch.cuda.stream(endpoint.stream):
        tensors = endpoint.video.proxy.uf_state_snapshot()
    endpoint.stream.synchronize()
    return tensors


def feed_base(codec, decoder, family):
    receiver = codec.receiver(decoder)
    receiver.feed(family.raw[:family.base_end // 2])
    if receiver.decode_available() != 0:
        raise RuntimeError("partial base was committed")
    receiver.feed(family.raw[family.base_end // 2:family.base_end])
    if receiver.decode_available() != 1 or receiver.decode_available() != 0:
        raise RuntimeError("base was not committed exactly once")
    return receiver


def feed_enhancement(receiver, family, index, budget):
    prefix, records = family.prefix(allowance(budget))
    start = family.base_end
    if index % 9 == 0:
        return 0, 0, 0.0, 0.0
    end = len(prefix) - 1 if index % 7 == 0 and records else len(prefix)
    begin = perf_counter()
    decoded = 0
    for stop in (start + (end - start) // 3, end):
        receiver.feed(prefix[start:stop])
        decoded += receiver.decode_available()
        start = stop
    latency = (perf_counter() - begin) * 1000
    expected = records - (end < len(prefix))
    if decoded != expected or receiver.decode_available():
        raise RuntimeError("a previous enhancement was replayed or a fragment decoded early")
    if end == len(prefix):
        receiver.finish()
    begin = perf_counter()
    output = receiver.render()
    torch.cuda.synchronize()
    if not torch.isfinite(output).all().item():
        raise RuntimeError("nonfinite enhanced display")
    return end - family.base_end, decoded, latency, (perf_counter() - begin) * 1000


def loss_case(codec, decoder, family, index):
    if index == 12:
        return "base_lost"
    receiver = codec.receiver(decoder)
    receiver.feed(family.raw[:family.base_end])
    try:
        receiver.decode_available()
    except WireError as e:
        if "DESYNC" not in str(e) or not 12 < index < 32:
            raise
        return "p_rejected"
    if 12 < index < 32:
        raise RuntimeError("prediction accepted after Base loss")
    return "independent_i_recovery" if index == 32 else "ok"


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--entropy-table", type=Path)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--sensitivity", type=Path)
    parser.add_argument("--low", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.method, args.precision, args.fixed = "d1", "bf16", False
    setup()
    reader, plans = inputs()
    assets = OfficialAssets.verify(args.weights)
    codec = load_codec(args, assets)
    base, dynamic, missing = [decoder_for(codec, assets) for _ in range(3)]
    probability = probability_evidence(codec, (base, dynamic, missing), args.entropy_table)
    rows, samples = [], []
    for clip in plans["long"]:
        codec.reset()
        for endpoint in (base, dynamic, missing):
            endpoint.reset()
        for index in range(clip.length):
            rgb = source_frame(reader, clip, index)
            encoded = encode(codec, rgb, index)
            family = encoded.family
            baseline = feed_base(codec, base, family)
            viewed = feed_base(codec, dynamic, family)
            display_budget = CYCLE[index % len(CYCLE)]
            if args.low and index % 4 == 0:
                low = (1, 2, 3)[(index // 4) % 3]
                if family.base_end <= allowance(low):
                    display_budget = low
            added, records, enhance_ms, render_ms = feed_enhancement(viewed, family, index, display_budget)
            if not torch.equal(baseline.render(), viewed.base) or not codec.matches(encoded, dynamic, viewed):
                raise RuntimeError("dynamic replay changed the exact Base output")
            intra = viewed.frame is not None and dynamic.order.last.intra
            if not intra and not all(torch.equal(a, b) for a, b in zip(snapshot(base), snapshot(dynamic), strict=True)):
                raise RuntimeError("dynamic replay changed the exact native DPB tensors")
            status = loss_case(codec, missing, family, index)
            if status in ("ok", "independent_i_recovery") and not torch.equal(
                dynamic.decoded_latent(), missing.decoded_latent()
            ):
                raise RuntimeError("independent I did not resynchronize native decoding")
            rows.append({"video": clip.video, "index": index, "type": "I" if intra else "P",
                         "base_bytes": family.base_end, "enhancement_delivered_bytes": added,
                         "records": records, "budget_mbps": display_budget, "loss_case": status,
                         "enhancement_decode_ms": enhance_ms, "render_ms": render_ms, "base_state_equal": True})
            if index in (0, 1):
                path = args.output / "samples" / f"{clip.video.split(':')[-1]}-{index}.huf"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(family.raw)
                samples.append({"path": path.name, "clip": clip.video, "index": index,
                                "sha256": hashlib.sha256(family.raw).hexdigest(), "boundaries": family.ends})
    write_csv(args.output / "frames.csv", rows)
    write_json(args.output / "samples/manifest.json", {"samples": samples})
    write_json(args.output / "summary.json", {
        "status": "PASS", "frames": len(rows), "state_mismatches": 0, "read_counts": reader.counts,
        "independent_i_recoveries": sum(row["loss_case"] == "independent_i_recovery" for row in rows),
        "rejected_p": sum(row["loss_case"] == "p_rejected" for row in rows), "long_lengths": [c.length for c in plans["long"]],
        "long_128_256": "NOT_AVAILABLE", "max_base": max(row["base_bytes"] for row in rows), "model_tag": codec.tag.hex(),
        "checkpoint_sha256": sha256_file(args.checkpoint) if args.checkpoint else None, **probability})


if __name__ == "__main__":
    main()
