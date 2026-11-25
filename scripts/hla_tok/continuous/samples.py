"""Export fixed-coordinate, native samples from the real BF16 wire path."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from hla_tok.evaluation.io import write_json
from hla_tok.evaluation.metrics import sha256_file
from hla_tok.evaluation.reconstruction import save_native
from hla_tok.streaming.continuous import encode_continuous, truncate
from scripts.hla_tok.continuous.inputs import RESULTS, V3, open_inputs
from scripts.hla_tok.continuous.matrix import Setting, compute, validation_frames
from scripts.hla_tok.continuous.ordering import load_orders
from scripts.hla_tok.continuous.wire_eval import build_wire_decoder

SIGN_CROP = (1320, 260, 384, 216)


def _image_evidence(folder: Path, label: str, vehicle_crop) -> dict:
    result = {}
    native = folder / f"{label}.png"
    with Image.open(native) as rgb:
        if rgb.size != (1920, 1080) or rgb.mode != "RGB":
            raise ValueError(f"sample resolution or RGB mode differs: {native}")
        result[native.name] = {"sha256": sha256_file(native), "width": 1920, "height": 1080}
        for suffix, (x, y, width, height) in (("crop", vehicle_crop), ("sign_crop", SIGN_CROP)):
            file = folder / f"{label}_{suffix}.png"
            crop = rgb.crop((x, y, x + width, y + height))
            if suffix == "sign_crop":
                crop.save(file)
            with Image.open(file) as recorded:
                if recorded.size != (width, height) or recorded.tobytes() != crop.tobytes():
                    raise ValueError(f"reported crop differs from native pixels: {file}")
            result[file.name] = {"sha256": sha256_file(file), "width": width, "height": height}
    return result


@torch.inference_mode()
def export_samples(v3: Path, output: Path) -> None:
    inputs = open_inputs(v3)
    freeze = inputs.config["sample"]
    video, frame, source = next(validation_frames(inputs, [(freeze["video"], freeze["frame"])]))
    folder = output / "samples"
    save_native(folder, "source", source)
    receiver = build_wire_decoder(inputs)
    names = ["source", "C1_fixed13_K16320"]
    packets = []
    with compute(inputs.config["quality_compute_dtype"]):
        full = encode_continuous(inputs.model, source)
        for order in load_orders(output):
            values = order.apply(full)
            for budget in freeze["budgets"]:
                label = f"{order.name.name}_bf16_K{budget}"
                image, size = receiver(Setting(order, budget, "bf16"), truncate(values, budget), frame)
                save_native(folder, label, image)
                names.append(label)
                packets.append({"sample": label, "budget": budget, "order": order.name.name} | size)
    files = {name: value for label in names for name, value in
             _image_evidence(folder, label, freeze["crop_xywh"]).items()}
    write_json(folder / "manifest.json", inputs.provenance() | {
        "video": video, "frame": frame, "sample_rule": freeze, "sign_crop_xywh": SIGN_CROP,
        "files": files, "wire_packets": packets,
        "existing_c1_reference": "c1_sample.json", "postprocessing": "RGB [0,1] to 8-bit PNG, same-coordinate crop",
    })
    inputs.finish(output / "samples_run.json")
    print(f"Recorded {len(files)} native images and matched crops from {video} frame {frame}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v3", type=Path, default=V3)
    parser.add_argument("--output", type=Path, default=RESULTS)
    args = parser.parse_args()
    export_samples(args.v3, args.output)


if __name__ == "__main__":
    main()
