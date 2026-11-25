"""Native one-pass evaluation of untrained unified latent refinement (D0)."""

import argparse
from pathlib import Path
from time import perf_counter

import torch
from torch import Tensor

from hla_tok.uf.anchor import MAX_APPLICATION_BYTES, FrameReceiver, Segmentation
from hla_tok.uf.latent import UnifiedCodec, UnifiedRefinement, observe
from hla_tok.uf.official import NativeDecoder, NativeEncoder, OfficialAssets
from hla_tok.uf.wire import PrefixFamily
from scripts.hla_tok.continuous.quality import NativeQuality
from scripts.hla_tok.uf.anchor_eval import Encoded, evaluate_encoder, locked_qp
from scripts.hla_tok.uf.inputs import OUTPUT, checkpoint_directory, inputs
from scripts.hla_tok.v2.protocol import setup


class UnifiedEncoder:
    def __init__(self, assets: OfficialAssets, codec: UnifiedCodec, qp: tuple[int, int]):
        self.native = NativeEncoder(assets)
        self.codec, self.qp = codec, qp

    def reset(self) -> None:
        self.native.reset()

    def receiver(self, decoder: NativeDecoder) -> FrameReceiver:
        return self.codec.receiver(decoder)

    def matches(self, encoded: Encoded, decoder: NativeDecoder, receiver: FrameReceiver) -> bool:
        return torch.equal(encoded.reference, decoder.decoded_latent())

    @torch.no_grad()
    def encode(self, rgb: Tensor, index: int, intra: bool, reset: bool = False) -> Encoded:
        start = perf_counter()
        base = self.native.encode(rgb, index, self.qp[0 if intra else 1], intra, reset)
        compressed = perf_counter()
        original, latent, native_step, *_ = observe(self.native, intra, "uf_encode_snapshot")
        analyzed = perf_counter()
        records = self.codec.records(intra, original, latent, native_step)
        family = PrefixFamily.build(base, self.codec.tag, records, MAX_APPLICATION_BYTES)
        packed = perf_counter()
        return Encoded(family, latent, {"native_encode_ms": 1e3 * (compressed - start),
                       "analysis_base_decode_ms": 0.0, "analysis_ms": 1e3 * (analyzed - compressed),
                       "quant_rank_entropy_pack_ms": 1e3 * (packed - analyzed),
                       "encode_ms": 1e3 * (packed - start)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--base-lock", type=Path, default=OUTPUT / "baseline/isolated/base_lock.json")
    parser.add_argument("--split", choices=("calibration", "validation", "long"), default="validation")
    parser.add_argument("--no-ranking", action="store_true")
    parser.add_argument("--static-cdf", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    setup()
    reader, plans = inputs()
    assets, qp = OfficialAssets.verify(checkpoint_directory(args.weights)), locked_qp(args.base_lock)
    torch.cuda.reset_peak_memory_stats()
    model = UnifiedRefinement().cuda().eval()
    codec = UnifiedCodec(model, Segmentation(8, 8, conditional=not args.static_cdf, ranked=not args.no_ranking))
    quality = NativeQuality()
    metadata = {"precision": "FP16 native and display; FP32 refinement arithmetic", "base_qp": qp,
                "model_tag": codec.tag.hex(), "optimizer_steps": 0,
                "enhancement_parameters": sum(parameter.numel() for parameter in model.parameters())}
    evaluate_encoder(assets, UnifiedEncoder(assets, codec, qp), reader, plans[args.split],
                     args.output, quality, metadata)


if __name__ == "__main__":
    main()
