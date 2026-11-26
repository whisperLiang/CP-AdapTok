"""Measure unquantized upstream HT-S y on the frozen official reference history."""

import argparse
from pathlib import Path

import torch
from torch import Tensor
from torch.nn import functional as F

from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.hts.native import NativeNetworks, frame_input, restore_native_rgb
from hla_tok.hts.units import NativeSchedule, Unit
from hla_tok.uf.official import NativeNAL
from scripts.hla_tok.continuous.quality import summarize
from scripts.hla_tok.hts.audit_native import CachedQuality, Point, encode_clip, source_clips
from scripts.hla_tok.uf.inputs import inputs
from third_party.DCVC.src.models.video_model_ht import DMC
from third_party.DCVC.src.utils.common import ModelStructure, get_state_dict, set_torch_env


def continuous_frames(net: DMC, rgb: Tensor, unit: Unit, context: Tensor, qp: int) -> tuple[Tensor, list[Tensor]]:
    height, width = rgb.shape[-2:]
    padded = F.pad(frame_input(rgb, unit).float(), (0, (-width) % 16, 0, (-height) % 16), mode="replicate")
    q_encoder = net.q_encoder[qp].reshape(1, -1, 1, 1)
    q_decoder = net.q_decoder[qp].reshape(1, -1, 1, 1)
    y = net.encoder(padded, context, q_encoder)
    frames, _ = net.get_recon_and_feature(y, context, q_decoder)
    return y, [tensor.clamp(-0.5, 0.5) for tensor in frames]


def probe(native, net, source, quality, qp):
    units = list(NativeSchedule(8, reset_interval=128).units(source.clip.length))
    stream, ledger = encode_clip(native, source, units, Point("official", qp, qp))
    nal = NativeNAL()
    quality.bind(source.rgb)
    base_metrics, continuous_metrics = [], []
    shape = None
    for unit, row in zip(units, ledger, strict=True):
        offset = row["offset"]
        payload = stream[offset:offset + row["total_bytes"]]
        native_frames = native.decode(nal.decode(payload, unit.start))
        if unit.intra:
            continue
        _, context = native.video.proxy.audit_context()
        before = native.reference_hash(unit)
        y, continuous = continuous_frames(net, source.rgb, unit, context.float(), qp)
        if native.reference_hash(unit) != before:
            raise ValueError("continuous decoder modified the official Base reference")
        shape = list(y.shape)
        for decoded, rows in ((native_frames, base_metrics), (continuous, continuous_metrics)):
            rgb = restore_native_rgb(decoded, unit, *source.rgb.shape[-2:])
            rows.extend(quality.score_at(index, frame) for index, frame in zip(unit.indices, rgb.split(1), strict=True))
    quality.references.clear()
    return [
        {"qp": qp, "sequence": source.clip.video, "representation": label, **summarize(rows),
         "p_latent_shape": shape, "measured_wire": label == "native",
         "p_native_bytes": sum(row["total_bytes"] for row in ledger if row["unit_type"] == "P")
         if label == "native" else None}
        for label, rows in (("native", base_metrics), ("continuous_y", continuous_metrics))
    ]


@torch.inference_mode()
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    set_torch_env()
    reader, plans = inputs()
    sources = source_clips(reader, plans["calibration"])
    metric = CachedQuality()
    rows = []
    with torch.cuda.stream(torch.cuda.Stream()):
        native = NativeNetworks(args.weights, "hts", 0.15)
        net = DMC(ModelStructure.HTS).eval().requires_grad_(False)
        net.load_state_dict(get_state_dict(args.weights / "cvpr2026_video_hts.pth.tar"), strict=True)
        net.cuda().to(memory_format=torch.channels_last)
        for source in sources:
            for qp in (0, 12, 32):
                rows.extend(probe(native, net, source, metric, qp))
    write_csv(args.output / "native_continuous_capacity.csv", rows)
    write_json(args.output / "native_continuous_scope.json", {
        "scope": "Calibration P frames at frozen native reference. Independent I initializes context.",
        "continuous_coding": "unquantized y tensor. No wire rate is assigned.",
        "pixels_read": reader.counts,
    })


if __name__ == "__main__":
    main()
