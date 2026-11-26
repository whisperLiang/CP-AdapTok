"""Replay exact HUF1 boundaries, RGB, DPB and frame availability with the chosen checkpoint."""

import argparse
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import torch

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.uf.anchor import BUDGETS
from hla_tok.uf.latent import UnifiedCodec
from hla_tok.uf.official import NativeDecoder, OfficialAssets, sha256_file
from hla_tok.uf.wire import APP_BASE_OVERHEAD
from hla_tok.uf_v2.native_weights import install_adjustments
from hla_tok.uf_v2.static_entropy import StaticAsset
from hla_tok.uf_v3.deploy import native_parameters
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.inputs import inputs
from scripts.hla_tok.uf.unified_eval import UnifiedEncoder
from scripts.hla_tok.uf_v2.dynamic import (
    CYCLE,
    feed_base,
    feed_enhancement,
    loss_case,
    probability_evidence,
    snapshot,
)
from scripts.hla_tok.uf_v2.evaluate import allowance, encode, receive
from scripts.hla_tok.uf_v2.incremental_bench import explicit_replay
from scripts.hla_tok.uf_v2.parity import parity
from scripts.hla_tok.uf_v3.audit import tensor_delta
from scripts.hla_tok.uf_v3.endpoint import load_endpoint, load_v2
from scripts.hla_tok.v2.protocol import setup


def prove_prefixes(codec, reader, clips, count_per_clip, out, table=None):
    base, dynamic, missing = [codec.new_decoder() for _ in range(3)]
    rows = []
    for clip in clips:
        codec.reset()
        for endpoint in (base, dynamic, missing):
            endpoint.reset()
        for index in range(min(clip.length, count_per_clip)):
            rgb = source_frame(reader, clip, index)
            encoded = encode(codec, rgb, index)
            family = encoded.family
            budget = CYCLE[index % len(CYCLE)]
            if index % 4 == 0:
                low = (1, 2, 3)[(index // 4) % 3]
                if family.base_end <= allowance(low):
                    budget = low
            ref, view = (feed_base(codec, endpoint, family) for endpoint in (base, dynamic))
            added, records, decode_ms, render_ms = feed_enhancement(view, family, index, budget)
            if not torch.equal(ref.render(), view.base) or not codec.matches(encoded, dynamic, view):
                raise RuntimeError("changed bandwidth modified decoded Base")
            intra = dynamic.order.last.intra
            if not intra and not all(torch.equal(a, b) for a, b in zip(snapshot(base), snapshot(dynamic), strict=True)):
                raise RuntimeError("display modified P frame reference tensors")
            earlier = family.raw[:family.base_end]
            for allowed in (allowance(2), *BUDGETS.values()):
                if allowed < family.base_end:
                    continue
                raw, count = family.prefix(allowed)
                if not raw.startswith(earlier) or raw != family.raw[:family.ends[count]] or len(raw) > allowed:
                    raise RuntimeError("nonphysical prefix or budget violation")
                earlier = raw
            lost = loss_case(codec, missing, family, index)
            if lost in ("ok", "independent_i_recovery") and not torch.equal(
                dynamic.decoded_latent(), missing.decoded_latent()
            ):
                raise RuntimeError("eligible reference did not resynchronize")
            rows.append({"video": clip.video, "index": index, "type": "I" if intra else "P", "base_bytes": family.base_end,
                         "budget": budget, "delivered_enhancement_bytes": added, "records": records,
                         "loss": lost, "enhancement_decode_ms": decode_ms, "render_ms": render_ms})
    write_csv(out / "dynamic.csv", rows)
    return {"frames": len(rows), "base_mismatches": 0, "max_base": max(row["base_bytes"] for row in rows),
            "i_recoveries": sum(row["loss"] == "independent_i_recovery" for row in rows),
            "p_rejected": sum(row["loss"] == "p_rejected" for row in rows),
            "long_128_256": "NOT_AVAILABLE", **probability_evidence(codec, (base, dynamic, missing), table)}


def every_budget_state(codec, reader, clip):
    budgets = (0, 2, 5, 10, 20, 40)
    codec.reset()
    peers = {budget: codec.new_decoder() for budget in budgets}
    records, skipped = 0, 0
    for index in range(34):
        with patch.object(codec.native, "encode", wraps=codec.native.encode) as native_encode:
            frame = encode(codec, source_frame(reader, clip, index), index)
        if native_encode.call_count != 1:
            raise RuntimeError("a full prefix family must analyze its current source exactly once")
        family = frame.family
        base_dp, base_rgb = None, None
        cuts = []
        for budget, decoder in peers.items():
            cap = allowance(budget) if budget else family.base_end
            decoded, stats = receive(codec, decoder, family, cap)
            if decoded is None:
                skipped += 1
                if stats["status"] not in ("BASE_BUDGET_EXCEEDED", "DESYNC"):
                    raise RuntimeError("unexpected budget replay discontinuity")
                continue
            receiver, rgb = decoded
            if not codec.matches(frame, decoder, receiver) or receiver.base is None:
                raise RuntimeError("native Base latent differs across budgets")
            if budget == 0:
                base_rgb = rgb
                base_dp = None if decoder.order.last.intra else snapshot(decoder)
            elif not torch.equal(base_rgb, receiver.base):
                raise RuntimeError("Base pixels differ across budgets")
            if base_dp is not None and any(not torch.equal(a, b) for a, b in zip(
                base_dp, snapshot(decoder), strict=True
            )):
                raise RuntimeError("native DPB differs across budgets")
            cuts.append(stats["bytes"])
            records += 1
        if cuts != sorted(cuts):
            raise RuntimeError("later budget produced a shorter prefix")
    return {"checked_frame_budgets": records, "skipped_unreachable": skipped, "reference_mismatches": 0,
            "base_and_p_dpb_exact": True, "frames": 34, "budgets": budgets, "one_native_encode_per_frame": True}


def fragmented(codec, reader, clip, out):
    full, incremental = codec.new_decoder(), codec.new_decoder()
    codec.reset()
    rows, renders = [], []
    for index in range(4):
        encoded = encode(codec, source_frame(reader, clip, index), index)
        family = encoded.family
        result, _ = receive(codec, full, family, BUDGETS[40])
        image, _, records, redraw = explicit_replay(codec, incremental, family, index)
        if result is None or not torch.equal(image, result[1]):
            raise RuntimeError("incremental reconstruction lost its exact physical prefix")
        rows.extend(records)
        renders.extend(redraw)
    write_csv(out / "append.csv", rows)
    write_csv(out / "render.csv", renders)
    return {"records": len(rows), "rgb_exact": True, "display_redraws": len(renders), "frames": 4}


def trajectory(codec, frames):
    codec.reset()
    decoder = codec.new_decoder()
    rows = []
    for index, rgb in enumerate(frames):
        encoded = encode(codec, rgb, index)
        family = encoded.family
        result, _ = receive(codec, decoder, family, BUDGETS[40])
        if result is None or not codec.matches(encoded, decoder, result[0]):
            raise RuntimeError("future-frame control failed to replay an independent prefix")
        state = () if decoder.order.last.intra else snapshot(decoder)
        rows.append({"wire": sha256(family.raw).hexdigest(),
                     "integers": sha256(result[0].canvas.symbols.tobytes()).hexdigest(),
                     "decoded_base": sha256(decoder.decoded_latent().cpu().contiguous().numpy().tobytes()).hexdigest(),
                     "base_dpb": [sha256(tensor.cpu().contiguous().numpy().tobytes()).hexdigest()
                                  for tensor in state]})
    return rows


def check_future_invariance(left, right):
    if left[:2] != right[:2] or left[2]["wire"] == right[2]["wire"]:
        raise RuntimeError("prefix of the sequence depends on future input")


def future_mutation(codec, reader, clip):
    images = [source_frame(reader, clip, index) for index in range(4)]
    changed = [*images[:2], *(image.flip(-1).mul(.5) for image in images[2:])]
    clean = [trajectory(codec, frames) for frames in (images, changed)]
    check_future_invariance(*clean)
    contaminated = []
    for frames in (images, changed):
        leaked = frames[0].mul(.875).add(frames[2], alpha=.125)
        contaminated.append(trajectory(codec, [leaked, *frames[1:]]))
    try:
        check_future_invariance(*contaminated)
    except RuntimeError as error:
        if "depends on future input" not in str(error):
            raise
    else:
        raise RuntimeError("future-invariance check missed a real lookahead injection")
    return {"unchanged_prefix_frames": 2, "modified_future_frames": 2, "sha256": clean,
            "future_leak_mutation": "KILLED", "mutated_prefix_sha256": [rows[0]["wire"] for rows in contaminated]}


def d0_equivalence(assets, candidate, reader, clip):
    fast = load_v2(assets, candidate, d0=True)
    old = UnifiedEncoder(assets, UnifiedCodec(fast.model), fast.qp)
    old.codec.tag = fast.tag
    install_adjustments(old.native, fast.adjustments, fast.qp[1])

    def independent(assets_arg):
        decoder = NativeDecoder(assets_arg)
        install_adjustments(decoder, fast.adjustments, fast.qp[1])
        return decoder

    # The original parity runner uses an official no-adjustment decoder by default.
    # Supply the matching Seed 2 Base transform to both endpoints.
    with patch("scripts.hla_tok.uf_v2.parity.NativeDecoder", independent):
        result = parity(assets, old, fast, reader, clip, 34)
    return result | {"checkpoint_sha256": sha256_file(candidate)}


def deployment(assets, endpoint, init, checkpoint, reader, clip):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "native_p" not in state:
        return {"status": "NOT_RUN"}
    previous = load_v2(assets, init)
    receiver = endpoint.new_decoder()
    matches = {}
    for kind, get in (("native_i", lambda ep: ep.intra), ("native_p", lambda ep: ep._video())):
        previous_weights = native_parameters(get(previous.native), kind)
        encoder_weights = native_parameters(get(endpoint.native), kind)
        decoder_weights = native_parameters(get(receiver), kind)
        for name, weight in encoder_weights.items():
            expected = state[kind][name].half()
            if not torch.equal(expected, weight.cpu()) or not torch.equal(expected, decoder_weights[name].cpu()):
                raise ValueError("checkpoint tensor differs from a physical encode/decode network")
            matches[f"{kind}.{name}"] = tensor_delta(previous_weights[name], weight)
    stale = load_endpoint(assets, checkpoint)
    old = previous.native._video()
    stale.native.stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stale.native.stream):
        stale.native._video().proxy.set_param(old.add_cdf_to_state_dict(old.state_dict()), old.gaussian_encoder.skip_thres)
    stale.native.stream.synchronize()
    endpoint.reset()
    previous.reset()
    evidence = []
    for index in range(4):
        rgb = source_frame(reader, clip, index)
        a, b, c = (encode(codec, rgb, index) for codec in (endpoint, previous, stale))
        out, _ = receive(endpoint, receiver, a.family, BUDGETS[40])
        if out is None or not endpoint.matches(a, receiver, out[0]):
            raise ValueError("re-exported weights did not agree on the independent decoder")
        same_bytes = a.family.raw[APP_BASE_OVERHEAD:] == b.family.raw[APP_BASE_OVERHEAD:]
        evidence.append({"index": index, "same_native_and_enhancement_payload": same_bytes,
                         "same_y": torch.equal(a.reference, b.reference),
                         "stale_cuda_detected": index > 0 and not torch.equal(a.reference, c.reference)})
    if all(row["same_y"] for row in evidence):
        return {"status": "UNVERIFIED", "fold": matches, "changes": evidence}
    stale_detected = any(row["stale_cuda_detected"] for row in evidence)
    if not stale_detected:
        raise RuntimeError("trained P Encoder did not affect the C++ CUDA path")
    return {"status": "PASS", "fold": matches, "changes": evidence,
            "checkpoint_sha256": sha256_file(checkpoint), "old_checkpoint_sha256": sha256_file(init)}


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--init", type=Path)
    parser.add_argument("--static", type=Path)
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--d0-equivalence", action="store_true")
    args = parser.parse_args()
    setup()
    reader, plans = inputs()
    assets = OfficialAssets.verify(args.weights)
    if args.d0_equivalence:
        write_json(args.output / "d0_parity.json", d0_equivalence(assets, args.checkpoint, reader, plans["validation"][1]))
        return
    codec = load_endpoint(assets, args.checkpoint, static=StaticAsset.load(args.static) if args.static else None,
                          raw=args.raw)
    summary = {"status": "PASS", "dynamic": prove_prefixes(codec, reader, plans["long"], 10000, args.output, args.static),
               "all_budgets": every_budget_state(codec, reader, plans["validation"][1]),
               "incremental": fragmented(codec, reader, plans["validation"][0], args.output),
               "future": future_mutation(codec, reader, plans["fit"][0])}
    if args.init:
        summary["deployment"] = deployment(assets, codec, args.init, args.checkpoint, reader, plans["fit"][0])
    write_json(args.output / "summary.json", summary | {"reads": reader.counts, "test_pixel_reads": reader.counts["test"]})


if __name__ == "__main__":
    main()
