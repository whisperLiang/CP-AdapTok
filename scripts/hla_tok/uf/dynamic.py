"""Check real changing-bandwidth prefixes across the archived native long clips."""

import argparse
from itertools import cycle
from pathlib import Path
from random import Random

import torch
from PIL import Image
from torch import Tensor

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.evaluation.metrics import psnr
from hla_tok.uf.anchor import BUDGETS, AnchorModel, FrameReceiver, RefinementCodec
from hla_tok.uf.base import schedule
from hla_tok.uf.latent import observe
from hla_tok.uf.official import NativeDecoder, OfficialAssets, sha256_file
from hla_tok.uf.wire import PrefixFamily
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.anchor_eval import AnchorEncoder, locked_qp
from scripts.hla_tok.uf.baseline import BASE_CAP
from scripts.hla_tok.uf.inputs import OUTPUT, checkpoint_directory, inputs
from scripts.hla_tok.v2.protocol import setup


def _fragment(receiver: FrameReceiver, packet: bytes, random: Random) -> None:
    start = 0
    while start < len(packet):
        end = start + random.randrange(1, 4097)
        receiver.feed(packet[start:end])
        receiver.decode_available()
        start = end
    receiver.finish()


def _same_reference(steady: NativeDecoder, varying: NativeDecoder, intra: bool) -> bool:
    if intra:
        return torch.equal(steady.decoded_latent(), varying.decoded_latent())
    first = observe(steady, False, "uf_state_snapshot")
    second = observe(varying, False, "uf_state_snapshot")
    return all(torch.equal(expected, actual) for expected, actual in zip(first, second, strict=True))


def dynamic_summary(rows: list[dict], counts: dict[str, int]) -> dict:
    base_exceeded = sum(row["base_bytes"] > BASE_CAP for row in rows)
    packet_exceeded = sum(row["bytes"] > BUDGETS[row["budget_mbps"]] for row in rows)
    state_mismatches = sum(not row["state_exact"] for row in rows)
    lengths = {name: sum(row["video"] == name for row in rows) for name in dict.fromkeys(row["video"] for row in rows)}
    passed = bool(rows) and not (base_exceeded or packet_exceeded or state_mismatches or counts["test"])
    return {"status": "PASS" if passed else "FAIL", "frame_count": len(rows), "lengths": list(lengths.values()),
            "base_max_bytes": {kind: max(row["base_bytes"] for row in rows if row["type"] == kind)
                               for kind in sorted({row["type"] for row in rows})},
            "base_exceeded": base_exceeded, "packet_exceeded": packet_exceeded, "state_mismatches": state_mismatches,
            "read_counts": counts, "native_128_256": "NOT_AVAILABLE", "codec_only": True}


def _write_pixels(target: Path, rgb: Tensor) -> dict:
    array = (rgb[0].clamp(0, 1).mul(255).round().to(torch.uint8).permute(1, 2, 0).contiguous().cpu().numpy())
    Image.fromarray(array).save(target)
    return {"file": target.name, "sha256": sha256_file(target), "bytes": target.stat().st_size}


def _write_native_images(directory: Path, name: str, rgb: Tensor, receiver: FrameReceiver, family: PrefixFamily) -> list[dict]:
    directory.mkdir(parents=True, exist_ok=True)
    files = [_write_pixels(directory / f"{name}-source.png", rgb),
             _write_pixels(directory / f"{name}-base.png", receiver.render())]
    cursor = family.base_end
    for budget, cap in BUDGETS.items():
        packet, records = family.prefix(cap)
        receiver.feed(packet[cursor:])
        receiver.decode_available()
        receiver.finish()
        files.append(_write_pixels(directory / f"{name}-{budget}.png", receiver.render())
                     | {"actual_application_bytes": len(packet), "records": records})
        cursor = len(packet)
    return files


@torch.no_grad()
def run(assets: OfficialAssets, qp: tuple[int, int], checkpoint: Path, output: Path, native_images: bool = False) -> None:
    setup()
    reader, plans = inputs()
    model = AnchorModel().cuda().eval()
    model.load_state_dict(torch.load(checkpoint, weights_only=True)["model"])
    encoder = AnchorEncoder(assets, RefinementCodec(model, "bf16"), qp)
    steady, varying = NativeDecoder(assets), NativeDecoder(assets)
    random = Random(197)
    rows, sample_hashes = [], []
    samples = output / "samples"
    samples.mkdir(parents=True, exist_ok=True)
    for clip in plans["long"]:
        manifest = clip.manifest(reader)["frames"]
        for endpoint in (encoder, steady, varying):
            endpoint.reset()
        for index, budget in zip(range(clip.length), cycle((5, 40, 10, 20, 5))):
            intra, reset = schedule(index)
            rgb = source_frame(reader, clip, index)
            encoded = encoder.encode(rgb, index, intra, reset)
            family = encoded.family
            base_only, _ = family.prefix(family.base_end)
            packet, records = family.prefix(BUDGETS[budget])
            before, after = encoder.receiver(steady), encoder.receiver(varying)
            _fragment(before, base_only, random)
            _fragment(after, packet, random)
            base, display = before.render(), after.render()
            finite = bool(torch.isfinite(display).all())
            exact = torch.equal(base, after.base) if after.base is not None else False
            state = finite and exact and _same_reference(steady, varying, intra)
            base_mse = float((base.float() - rgb).square().mean())
            display_mse = float((display.float() - rgb).square().mean())
            rows.append({"video": clip.video, "source_index": clip.start + index, "frame": index,
                         "type": "I" if intra else "P", "budget_mbps": budget, "bytes": len(packet),
                         "base_bytes": family.base_end, "base_cap_pass": family.base_end <= BASE_CAP,
                         "records": records, "state_exact": state, "base_mse": base_mse, "display_mse": display_mse,
                         "base_psnr": psnr(base_mse), "display_psnr": psnr(display_mse)})
            if index in (0, 1):
                name = f"{clip.video.split(':')[-1]}-{index}"
                sample = samples / f"{name}.huf"
                sample.write_bytes(family.raw)
                sample_hashes.append({"video": clip.video, "source_index": clip.start + index, "file": sample.name,
                                      "source_sha256": manifest[index]["sha256"],
                                      "sha256": sha256_file(sample), "bytes": len(family.raw),
                                      "prefix_ends": list(family.ends), "pixels_hw": [1080, 1920]})
                if native_images and index == 1 and len(sample_hashes) == 2:
                    sample_hashes[-1]["native_images"] = _write_native_images(output / "native", name, rgb, before, family)
    write_csv(output / "frames.csv", rows)
    write_json(samples / "manifest.json", {"samples": sample_hashes, "checkpoint_sha256": sha256_file(checkpoint)})
    write_json(output / "summary.json", dynamic_summary(rows, reader.counts))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--base-lock", type=Path, default=OUTPUT / "baseline/isolated/base_lock.json")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--native-images", action="store_true")
    args = parser.parse_args()
    assets = OfficialAssets.verify(checkpoint_directory(args.weights))
    run(assets, locked_qp(args.base_lock), args.checkpoint, args.output, native_images=args.native_images)


if __name__ == "__main__":
    main()
