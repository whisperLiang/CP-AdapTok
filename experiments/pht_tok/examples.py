"""Export a small independent Calibration I+P stream and decoded image crops."""

import argparse
from pathlib import Path

import torch
from PIL import Image

from experiments.pht_tok.study import ROOT, SKIP, calibration, write_json
from hla_tok.hts.native import NativeNetworks
from pht_tok.bitstream import Mode
from pht_tok.codec import PHTDecoder, PHTEncoder, model_identifier
from third_party.DCVC.src.utils.common import set_torch_env
from third_party.DCVC.src.utils.transforms import ycbcr2rgb

CROP = (720, 368, 1200, 638)


def save(img, folder: Path, stem: str):
    rgb = img[0].permute(1, 2, 0).clamp(0, 1)
    pil = Image.fromarray(rgb.mul(255).round().byte().cpu().numpy())
    pil.save(folder / f"{stem}.jpg", quality=94)
    pil.crop(CROP).save(folder / f"{stem}-crop.png")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "results/pht_tok/samples")
    args = parser.parse_args()
    args.output.mkdir(exist_ok=True, parents=True)
    set_torch_env()
    with torch.cuda.stream(torch.cuda.Stream()):
        sources, counts = calibration()
        source = sources[0]
        ident = model_identifier(args.weights, SKIP)
        tx = PHTEncoder(NativeNetworks(args.weights, "hts", SKIP), ident)
        rx = PHTDecoder(NativeNetworks(args.weights, "hts", SKIP), ident)
        save(source.rgb[1:2], args.output, "source")
        entries = {}
        for mode in (Mode.MASK, Mode.PRECISION):
            tx.key = rx.key = None
            anchor = tx.encode_intra(source.yuv[:1], 32, 0)
            rx.decode_prefix(anchor.data)
            chunk = tx.encode_chunk_once(source.yuv[1:9], 32, 1, 2, mode=mode)
            key = mode.name.lower()
            (args.output / "anchor.pht").write_bytes(anchor.data)
            (args.output / f"{key}.pht").write_bytes(chunk.data)
            entries[key] = chunk.ledger
            session = rx.begin_chunk()
            previous = 0
            for stage, boundary in enumerate(chunk.prefix_boundaries, 1):
                session.append_bytes(chunk.data[previous:boundary])
                session.decode_new_layers()
                if mode == Mode.MASK or stage in (1, 4):
                    save(ycbcr2rgb(session.reconstruct()[0] + 0.5), args.output, f"{key}-{stage}")
                previous = boundary
        if rx.adapter.reference.digest() != tx.adapter.reference.digest():
            raise ValueError("sample's shared reference differs between endpoints")
    write_json(args.output / "manifest.json", {
        "source": source.name, "split": "Train Calibration", "qp": 32, "k_ref": 2,
        "source_original_indices": list(range(29, 38)), "display_source_index": 30,
        "resolution": [1080, 1920], "crop_box_left_top_right_bottom": CROP,
        "crop_encoding": "lossless PNG from the directly reconstructed RGB8 image",
        "full_image_encoding": "JPEG quality 94", "streams": entries, "pixel_reads": counts
    })


if __name__ == "__main__":
    main()
