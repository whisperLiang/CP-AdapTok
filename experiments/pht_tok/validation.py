"""One frozen, closed-loop Validation replay without candidate selection."""

import argparse
from functools import partial
from pathlib import Path

import torch

from baselines.dcvc_uf.native import NativeNetworks
from baselines.dcvc_uf.official import NativeNAL
from baselines.dcvc_uf.units import NativeSchedule
from common.artifacts import write_json
from common.inputs import inputs, source_clips
from common.performance import timed
from experiments.pht_tok.closed import official_encode
from experiments.pht_tok.study import ROOT, SKIP, Quality, Source, aggregate_curves, ledger_row
from pht_tok.bitstream import Mode
from pht_tok.codec import PHTDecoder, PHTEncoder, model_identifier
from third_party.DCVC.src.utils.common import set_torch_env
from third_party.DCVC.src.utils.transforms import rgb2ycbcr

QP = 32
REFERENCE_PREFIX = 2


def run_clip(source, official, tx, rx, metrics):
    units = list(NativeSchedule(8, reset_interval=128).units(len(source.rgb)))
    writer, reader = NativeNAL(), NativeNAL()
    control = [official_encode(official, source, unit, QP, writer) for unit in units]
    tx.key = rx.key = None
    anchor = tx.encode_intra(source.yuv[:1], QP, 0)
    anchor_session = rx.decode_prefix(anchor.data)
    rows = []
    for unit, packet in zip(units, control, strict=True):
        raw = source.rgb[unit.start:unit.start + unit.valid]
        native = official.decode(reader.decode(packet, unit.start))
        score = metrics.score(raw, native[:unit.valid])
        rows.append(ledger_row(source, unit, QP, "official", 4, len(packet), score))
        if unit.intra:
            anchor_score = metrics.score(raw, anchor_session.reconstruct())
            rows.extend(ledger_row(source, unit, QP, "MASK_R1", stage, len(anchor.data), anchor_score)
                        for stage in range(REFERENCE_PREFIX, 5))
            continue
        valid = source.yuv[unit.start:unit.start + unit.valid]
        chunk, encode_ms = timed(partial(tx.encode_chunk_once, mode=Mode.MASK, reset=unit.reset),
                                 valid, QP, unit.start, REFERENCE_PREFIX)
        session = rx.begin_chunk()
        previous = 0
        for stage, boundary in enumerate(chunk.prefix_boundaries, 1):
            if stage < REFERENCE_PREFIX:
                continue
            session.append_bytes(chunk.data[previous:boundary])
            session.decode_new_layers()
            output = session.reconstruct()
            row = ledger_row(source, unit, QP, "MASK_R1", stage, boundary, metrics.score(raw, output))
            rows.append(row | {"encode_ms": encode_ms, "hash": rx.adapter.reference.digest()})
            previous = boundary
        if tx.adapter.reference.digest() != rx.adapter.reference.digest():
            raise ValueError("Validation reference state drift")
    return rows


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--work", type=Path, default=ROOT / ".build/pht-tok")
    args = parser.parse_args()
    set_torch_env()
    with torch.cuda.stream(torch.cuda.Stream()):
        reader, plans = inputs()
        data = source_clips(reader, plans["validation"])
        if reader.counts != {"train": 0, "validation": 174, "test": 0}:
            raise ValueError("Validation must touch exactly the frozen cohort")
        model = model_identifier(args.weights, SKIP)
        tx = PHTEncoder(NativeNetworks(args.weights, "hts", SKIP), model)
        rx = PHTDecoder(NativeNetworks(args.weights, "hts", SKIP), model)
        official = NativeNetworks(args.weights, "hts", SKIP)
        metrics = Quality()
        rows = []
        for item in data:
            source = Source(item.clip.video, item.fps, item.rgb, rgb2ycbcr(item.rgb).half() - 0.5)
            rows.extend(run_clip(source, official, tx, rx, metrics))
            print("Validation", source.name, flush=True)
    write_json(args.work / "validation.json", {"frames": reader.counts, "qp": QP,
               "reference_prefix": REFERENCE_PREFIX, "curves": aggregate_curves(rows), "units": rows,
               "test_admission": "no authorization to open Test"})


if __name__ == "__main__":
    main()
