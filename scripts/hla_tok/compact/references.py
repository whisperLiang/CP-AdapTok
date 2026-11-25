"""Matched legacy continuous, C1 and Original AdapTok physical-byte comparisons."""

import subprocess
from pathlib import Path
from time import perf_counter

import torch

from experiments.original_adaptok_1080p.tiling import Stitcher, TileLayout, extract_batches
from hla_tok.evaluation.io import timed_cuda, write_csv, write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.evaluation.official import CLIP_LENGTH, OfficialBaseline, source_windows
from hla_tok.evaluation.reconstruction import save_native
from hla_tok.streaming.continuous import MAX_K, NATIVE, ContinuousReceiver, encode_continuous
from hla_tok.streaming.continuous_wire import (
    WireDtype,
    deserialize_continuous_tokens,
    max_budget_for_bytes,
    serialize_continuous_tokens,
    wire_rate,
)
from hla_tok.streaming.fixed13 import pack_tokens_13bit, unpack_tokens_13bit
from scripts.hla_tok.compact.fixed_reference import HEADER_BYTES, FixedWindow, open_window
from scripts.hla_tok.compact.inputs import OUTPUT, compact_inputs, compact_orders
from scripts.hla_tok.continuous.inputs import (
    Inputs,
    load_model,
    locked_output,
    run_arguments,
    verify_model_state,
)
from scripts.hla_tok.continuous.matrix import compute, groups, validation_frames
from scripts.hla_tok.continuous.quality import NativeQuality, summarize
from scripts.hla_tok.v2.protocol import ROOT, csv_at, json_at

OFFICIAL_CHECKPOINT = ROOT / "results/original_adaptok_1080p/checkpoint_manifest.json"


def _legacy_image(receiver, ordered, order, frame, count):
    raw = serialize_continuous_tokens(ordered[:, :count], WireDtype.BF16, frame, order=order.name)
    packet = deserialize_continuous_tokens(raw)
    return receiver.decode(packet.values.cuda(), order), len(raw), len(raw) - count * 64


@torch.inference_mode()
def single_frame_references(inputs: Inputs, config: dict, v3: Path, output: Path) -> None:
    c1, checkpoint = load_model(v3, "C1", inputs.config)
    decoder = c1.receiver()
    quality = NativeQuality()
    order = compact_orders()["O2"]
    receiver = ContinuousReceiver.from_model(inputs.model)
    caps = dict(zip(config["bandwidth_mbps"], config["frame_wire_byte_caps"], strict=True))
    cases = {cap: max_budget_for_bytes(cap, WireDtype.BF16) for cap in caps.values()}
    cases[0] = MAX_K
    equal_payload = config["equal_payload"]["cap_bytes"]
    cases[-equal_payload] = max_budget_for_bytes(equal_payload, WireDtype.BF16, include_header=False)
    rows = []
    selection = [(video, i) for video, n in inputs.cohort.items() for i in range(n)]
    sample = config["sample"]
    for video, frame, original in validation_frames(inputs, selection):
        quality.prepare(original)
        with compute(config["compute_dtype"]):
            native = order.apply(encode_continuous(inputs.model, original))
            enc = c1.encode_training(original, MAX_K, quantize=True)
            if enc.ids is None:
                raise ValueError("C1 has no Fixed13 indices")
            packet = FixedWindow(frame, 1, 1, 1, MAX_K).packet(pack_tokens_13bit(enc.ids[0].cpu().tolist()))
            _, received = open_window(packet)
            c1_rgb = decoder.decode_bytes(received, MAX_K, NATIVE)
        if len(received) != equal_payload:
            raise ValueError("C1 physical payload differs from the frozen endpoint")
        tags = {"video": video, "frame": frame}
        rows.append(tags | {"method": "C1 Fixed13", "cap_bytes": -equal_payload, "budget": MAX_K,
                            "payload_bytes": len(received), "total_bytes": len(packet)} | quality.score(c1_rgb))
        save = (video, frame) == (sample["video"], sample["frame"])
        if save:
            save_native(output / "samples", "C1_Fixed13", c1_rgb)
        for cap, k in cases.items():
            with compute(config["compute_dtype"]):
                image, size, head = _legacy_image(receiver, native, order, frame, k)
            rows.append(tags | {"method": "C0 BF16", "cap_bytes": cap, "budget": k,
                                "payload_bytes": size - head, "total_bytes": size} | quality.score(image))
            if save and (not cap or cap == caps[5]):
                save_native(output / "samples", "C0_BF16_K16320" if not cap else "C0_BF16_5Mbps", image)
    write_csv(output / "single_frame_references_per_frame.csv", rows)
    summary = []
    for keys, items in groups(rows, ("method", "cap_bytes", "budget")):
        p, total = (sum(row[key] for row in items) for key in ("payload_bytes", "total_bytes"))
        summary.append(keys | summarize(items) | wire_rate(p, total, len(items)))
    write_csv(output / "single_frame_references.csv", summary)
    verify_model_state(c1, checkpoint["state_dict_sha256"])
    write_json(output / "c1_reference_identity.json", {
        "published_c1_checkpoint": checkpoint, "total_wire_scope": "independent F13W evaluation envelope",
        "historical_full_header_comparability": "NOT_COMPARABLE", "new_header_bytes": HEADER_BYTES,
    })


def verify_archive(inputs: Inputs, config: dict) -> dict:
    commit = config["reference"]["original_commit"]
    observed = subprocess.check_output(["git", "rev-parse", "experiment/original-adaptok-1080p"], cwd=ROOT, text=True)
    if observed.strip() != commit:
        raise ValueError("frozen Original AdapTok Git reference differs")
    branch_record = subprocess.check_output(
        ["git", "show", f"{commit}:results/original_adaptok_1080p/checkpoint_manifest.json"], cwd=ROOT)
    if branch_record != OFFICIAL_CHECKPOINT.read_bytes():
        raise ValueError("Original AdapTok archived checkpoint manifest differs")
    folder = ROOT / "results/hla_tok/baselines"
    weights = json_at(folder / "official_weights.json")
    manifest = json_at(OFFICIAL_CHECKPOINT)
    if (weights["metrics"] != inputs.metric_weights
            or weights["weights_sha256_after"] != manifest["model_safetensors_sha256"]):
        raise ValueError("archived Original AdapTok used a different model or metric")
    paths = [folder / "stride128_bf16" / filename for filename in ("per_frame.csv", "windows.csv")]
    rows, windows = map(csv_at, paths)
    budgets = sorted({int(row["tokens_per_tile_block"]) for row in windows})
    expected = {(v, i, k) for v, n in inputs.cohort.items() for i in range(n) for k in budgets}
    observed_keys = {(r["video"], int(r["frame"]), int(r["tokens_per_tile_block"])) for r in rows}
    if expected != observed_keys or len(rows) != len(expected):
        raise ValueError("Original AdapTok frame identities or budget rows differ")
    return {"archive_commit": commit, "archive_sha256": {p.name: sha256_file(p) for p in paths},
            "archive_weights_sha256": sha256_file(folder / "official_weights.json"), "cohort": inputs.cohort,
            "cohort_sha256": inputs.config["cohort_sha256"], "budgets_per_group": budgets,
            "weights": weights, "status": "PASS", "rate_scope": "archived actual packed Fixed13 payload"}


def _official_encode(baseline, target, layout, batch_size):
    return torch.cat([baseline.encode(tiles)[0] for _, tiles in extract_batches(target, layout, batch_size)])


def _official_decode(baseline, received, coordinates, layout, k, batch_size):
    writer = Stitcher(layout, CLIP_LENGTH, 3, received.device)
    for offset in range(0, len(received), batch_size):
        stop = offset + batch_size
        writer.add(coordinates[offset:stop], baseline.decode(received[offset:stop], k))
    return writer.finish()


def _official_packet(ids, start, valid, k):
    selected = ids[..., :k].contiguous().cpu()
    size, groups, count = selected.shape
    header = FixedWindow(start, valid, size, groups, count)
    payload = pack_tokens_13bit(selected.flatten().tolist())
    raw = header.packet(payload)
    parsed, buffer = open_window(raw)
    decoded = torch.tensor(unpack_tokens_13bit(buffer, parsed.tokens)).reshape(selected.shape)
    if not torch.equal(selected, decoded):
        raise ValueError("Original AdapTok physical indices changed")
    return raw, decoded.to(ids.device), len(buffer)


def _official_window(baseline, clip, start, valid, reference, budgets):
    target = clip.cuda().float().div_(255)
    stride, batch_size = reference["original_stride"], reference["original_tile_batch"]
    layout = TileLayout(1080, 1920, 128, stride)
    coordinates = list(layout.coordinates)
    ids, enc_ms, _ = timed_cuda(lambda: _official_encode(baseline, target, layout, batch_size))
    output = []
    for k in budgets:
        (packet, received, length), wire_ms, _ = timed_cuda(lambda: _official_packet(ids, start, valid, k))
        decoded, dec_ms, _ = timed_cuda(lambda: _official_decode(baseline, received, coordinates, layout, k, batch_size))
        window = {"budget_per_group": k, "tokens_per_tile": k * ids.shape[1], "window_start": start,
                  "valid_frames": valid, "padded_frames": CLIP_LENGTH - valid, "tile_count": len(coordinates),
                  "payload_bytes": length, "header_bytes": len(packet) - length, "total_bytes": len(packet),
                  "encode_ms": enc_ms, "decode_and_stitch_ms": dec_ms, "pack_and_transfer_ms": wire_ms}
        output.append((decoded, window))
    return target, output


def _official_summary(rows: list[dict], windows: list[dict]) -> list[dict]:
    report = []
    for keys, data in groups(rows, ("budget_per_group", "tokens_per_tile")):
        matched = [w for w in windows if w["budget_per_group"] == keys["budget_per_group"]]
        payload, total = (sum(w[name] for w in matched) for name in ("payload_bytes", "total_bytes"))
        ms = sum(w["encode_ms"] + w["decode_and_stitch_ms"] + w["pack_and_transfer_ms"] for w in matched)
        report.append(keys | summarize(data) | wire_rate(payload, total, len(data)) |
                      {"method": "Original AdapTok", "window_frames": CLIP_LENGTH, "windows": len(matched),
                       "valid_source_frames": len(data), "codec_fps": 1000 * len(data) / ms,
                       "buffer_frames_before_first_output": CLIP_LENGTH - 1,
                       "buffer_ms_at_30fps": 1000 * (CLIP_LENGTH - 1) / 30,
                       "rate_scope": "F13W window envelope per valid source frame, including padded tails"})
    return report


@torch.inference_mode()
def original_reference(inputs: Inputs, config: dict, output: Path, checkpoint: Path | None) -> None:
    archive = verify_archive(inputs, config)
    write_json(output / "original_archive_identity.json", archive)
    manifest = json_at(OFFICIAL_CHECKPOINT)
    path = Path(manifest["directory"]) if checkpoint is None else checkpoint
    model = OfficialBaseline(ROOT / "third_party/AdapTok", path, manifest)
    checkpoint_config = manifest["config"]
    write_json(output / "original_temporal_scope.json", {
        "api_input_frames": CLIP_LENGTH, "encoder_temporal_patch_frames": model.model.temporal_patch_size,
        "decoder_temporal_patch_frames": model.model.decoder_temporal_patch_size,
        "temporal_groups": model.groups, "encoder_attention": checkpoint_config["encoder_attn_type"],
        "decoder_attention": checkpoint_config["decoder_attn_type"], "cross_window_state": False,
        "frames_waited_after_first_input": CLIP_LENGTH - 1,
        "buffer_ms_at_30fps": (CLIP_LENGTH - 1) * 1000 / 30,
        "scope": "entire 16-frame input required by the measured evaluation API",
    })
    quality = NativeQuality()
    reference = config["reference"]
    budgets = sorted({k if k <= model.ranks else k // model.groups for k in reference["original_budgets"]})
    rows: list[dict] = []
    windows: list[dict] = []
    sample = config["sample"]
    for video, start, valid, clip in source_windows(inputs.reader, inputs.cohort):
        begin = perf_counter()
        with compute(config["compute_dtype"]):
            target, decoded = _official_window(model, clip, start, valid, reference, budgets)
        for index in range(valid):
            quality.prepare(target[index:index + 1])
            for rgb, entry in decoded:
                tags = {key: entry[key] for key in ("budget_per_group", "tokens_per_tile")}
                rows.append(tags | {"video": video, "frame": start + index} | quality.score(rgb[index:index + 1]))
                if (video, start + index) == (sample["video"], sample["frame"]) and entry["budget_per_group"] == model.ranks:
                    save_native(output / "samples", "Original_AdapTok_K2048_per_tile", rgb[index:index + 1])
        windows.extend({"video": video} | item for _, item in decoded)
        print(f"Original AdapTok {video} {start}+{valid}: {perf_counter() - begin:.1f} s", flush=True)
    write_csv(output / "original_window_ledger.csv", windows)
    write_csv(output / "original_per_frame.csv", rows)
    write_csv(output / "original_rd.csv", _official_summary(rows, windows))
    if sha256_file(path / "model.safetensors") != manifest["model_safetensors_sha256"]:
        raise ValueError("Original AdapTok weights changed")


def main() -> None:
    parser = run_arguments(__doc__)
    parser.set_defaults(output=OUTPUT)
    parser.add_argument("stage", choices=("frame", "original"))
    parser.add_argument("--official-checkpoint", type=Path)
    args = parser.parse_args()
    with locked_output(args.output):
        inputs, config = compact_inputs(args.v3, args.output)
        if args.stage == "frame":
            single_frame_references(inputs, config, args.v3, args.output)
        else:
            original_reference(inputs, config, args.output, args.official_checkpoint)
        inputs.finish(args.output / f"{args.stage}_reference_inputs.json")


if __name__ == "__main__":
    main()
