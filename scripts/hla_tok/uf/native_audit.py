"""Record observed tensors from official I/P inference without changing reference state."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from hla_tok.evaluation.io import write_json
from hla_tok.uf.latent import observe
from hla_tok.uf.official import NativeDecoder, NativeEncoder, OfficialAssets, sha256_file
from scripts.hla_tok.temporal.inputs import source_frame
from scripts.hla_tok.uf.inputs import OUTPUT, checkpoint_directory, inputs
from scripts.hla_tok.v2.protocol import setup

ENCODED_TENSORS = {
    "I": ("y_normalized", "y_base", "q_step", "z_hat", "y_symbols_final_mask", "prior_final_mask"),
    "P": ("y_normalized", "y_base", "q_step", "z_hat", "y_symbols", "prior_common", "mean_second_mask"),
}
DECODER_TENSORS = {"I": ("y_base", "q_step"), "P": ("y_base", "q_step", "context", "base_feature")}
REFERENCE_TENSORS = ("memory", "context", "base_feature", "refresh_feature")


def statistics(value: Tensor) -> dict:
    numeric = value.float()
    return {"shape": list(value.shape), "dtype": str(value.dtype), "finite": bool(numeric.isfinite().all()),
            "minimum": float(numeric.min()), "maximum": float(numeric.max()),
            "mean_abs": float(numeric.abs().mean()), "zero_fraction": float((numeric == 0).float().mean())}


def symbol_entropy(value: Tensor) -> float:
    weights = np.unique(value.to(torch.int16).cpu().numpy(), return_counts=True)[1] / value.numel()
    return float(-(weights * np.log2(weights)).sum())


def verify_latent(encoder_tensors: list[Tensor], decoder_tensors: list[Tensor]) -> None:
    if not all(torch.equal(encoded, decoded)
               for encoded, decoded in zip(encoder_tensors[1:3], decoder_tensors[:2], strict=True)):
        raise ValueError("official encoded y_base/q_step differ from independently decoded tensors")


@torch.no_grad()
def run(weights: Path, output: Path) -> None:
    setup()
    reader, plans = inputs()
    clip = plans["calibration"][0]
    assets = OfficialAssets.verify(weights)
    encoder, decoder = NativeEncoder(assets), NativeDecoder(assets)
    frames = []
    for index in range(6):
        intra = index == 0
        kind = "I" if intra else "P"
        frame = encoder.encode(source_frame(reader, clip, index), index, 0 if intra else 8, intra)
        decoded = decoder.decode(frame)
        encoded_tensors = observe(encoder, intra, "uf_encode_snapshot")
        decoder_tensors = observe(decoder, intra, "uf_display_snapshot")
        verify_latent(encoded_tensors, decoder_tensors)
        state = {} if intra else dict(zip(REFERENCE_TENSORS, observe(decoder, False, "uf_state_snapshot"),
                                         strict=True))
        sample = dict(zip(ENCODED_TENSORS[kind], encoded_tensors, strict=True))
        sample.update(zip(DECODER_TENSORS[kind], decoder_tensors, strict=True))
        centered = sample["y_normalized"].float() - sample["y_base"].float() / sample["q_step"].float()
        frames.append({"index": index, "type": kind, "qp": frame.info.qp, "native_bytes": len(frame.payload),
                       "shape": list(decoded.shape), "encoded_decoder_equal": True,
                       "tensors": {name: statistics(value) for name, value in (sample | state).items()},
                       "normalized_residual": statistics(centered), "z_entropy_bits": symbol_entropy(sample["z_hat"]),
                       "y_symbol_entropy_bits": symbol_entropy(encoded_tensors[4])})
    write_json(output, {"clip": clip.video, "frames": frames, "read_counts": reader.counts,
                       "weights": {"I": sha256_file(assets.intra), "P": sha256_file(assets.ld)},
                       "z_byte_fraction": None, "z_byte_scope": "z and y share a native rANS payload",
                       "i_symbol_scope": "final of four native spatial-prior masks"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--output", type=Path, default=OUTPUT / "baseline/native_latents.json")
    args = parser.parse_args()
    run(checkpoint_directory(args.weights), args.output)


if __name__ == "__main__":
    main()
