"""Load a checkpoint before constructing the byte backend and the independent endpoints."""

import torch

from hla_tok.uf.anchor import Segmentation
from hla_tok.uf.base import BaseFrame
from hla_tok.uf.wire import FrameParser
from hla_tok.uf_v2.decoder import SharedDecoder, SharedRefinement
from hla_tok.uf_v2.native_weights import NativeAdjustments
from hla_tok.uf_v2.records import BatchRecords
from hla_tok.uf_v2.runtime import BASE_STANDARD_QP, Unified, UnifiedRefinement
from hla_tok.uf_v3.deploy import JointUnified

GEOMETRY = Segmentation(8, 8)


def load_endpoint(assets, checkpoint, *, d0=False, static=None, raw=False):
    weights = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = UnifiedRefinement().cuda().eval()
    model.load_state_dict(weights["quantizer"])
    display = None
    if not d0:
        display = SharedRefinement().cuda().half().to(memory_format=torch.channels_last).eval()
        display.load_state_dict(weights["refiner"])
    adjustment = NativeAdjustments().cuda().requires_grad_(False)
    adjustment.load_state_dict(weights["native_adjustments"])
    bank = None if static is None else BatchRecords(tuple(model.scales().tolist()), GEOMETRY, static.tables, raw)
    if "native_i" in weights:
        overlays = {kind: weights[kind] for kind in ("native_i", "native_p")}
        return JointUnified(assets, model, BASE_STANDARD_QP, display, bank=bank, adjustments=adjustment,
                            overlays=overlays)
    return Unified(assets, model, BASE_STANDARD_QP, display, bank=bank, adjustments=adjustment)


def load_v2(assets, checkpoint, **kwargs):
    return load_endpoint(assets, checkpoint, **kwargs)


def standard_decoder(codec) -> SharedDecoder:
    decoder = codec.new_decoder()
    if not isinstance(decoder, SharedDecoder):
        raise TypeError("matched diagnostics require the shared Base feature")
    return decoder

def parse_base(family, tag) -> BaseFrame:
    parser = FrameParser(tag)
    parser.feed(family.raw[:family.base_end])
    [frame] = parser.available()
    parser.finish()
    if not isinstance(frame, BaseFrame):
        raise ValueError("returned prefix did not contain Base")
    return frame
