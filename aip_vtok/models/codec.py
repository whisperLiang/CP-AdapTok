"""EncodeOnce and receiver sessions with fixed-prefix latent reference commits.

Gate 1 transports integer IDs in memory only. This API deliberately does not
claim a binary bitstream or bitrate until the Gate 2 entropy path is enabled.
"""

from dataclasses import dataclass

import torch
from torch import nn

from aip_vtok.backbones.latent_to_ref import LatentToRef
from aip_vtok.backbones.temporal_predictor import TemporalPredictor
from aip_vtok.decoder.fusion_decoder import FusionDecoder
from aip_vtok.decoder.latent_canvas import LatentCanvas
from aip_vtok.evaluation.evaluate import create_backbone, load_alignment
from aip_vtok.runtime import resolve
from aip_vtok.tokenizer.residual_tokenizer import ResidualTokenizer, TokenEncoding
from aip_vtok.tokenizer.tier_map import scan_plan


@dataclass
class EncodedFrame:
    frame_id: int
    tokens: TokenEncoding


class DecodeSession:
    def __init__(self, codec, state, frame_id, tiers):
        self.codec = codec
        self.context = codec.backbone.begin_frame(state, frame_id)
        device = codec.tokenizer.lifts[0].weight.device
        plan = scan_plan(tiers, codec.tokenizer.maximum).to(device)
        prediction = codec.predictor(state.latent, self.context.intra, plan.latent_shape, device)
        self.canvas = LatentCanvas(codec.tokenizer, plan, prediction)
        self.reference_budget = min(
            codec.reference_i if self.context.intra else codec.reference_p, len(plan.positions)
        )
        self.reference_latent = None
        self.rgb_decoder_calls = 0

    def append(self, values, offset, codes=False):
        count = values.shape[1]
        if offset != self.canvas.count or not 0 < count <= 128:
            raise ValueError("nonconsecutive or oversized append")
        split = self.reference_budget - offset
        pieces = [(values, offset)]
        if 0 < split < count:
            pieces = [(values[:, :split], offset), (values[:, split:], offset + split)]
        for part, position in pieces:
            if codes:
                self.canvas.append_codes(part, position)
            else:
                self.canvas.append_ids(part, position)
            if self.canvas.count == self.reference_budget:
                self.reference_latent = self.canvas.latent().clone()

    def render_frame(self, checkpointed=False):
        latent = self.codec.fusion(self.canvas.latent(), self.canvas.masks)
        rgb, _ = self.codec.backbone.decode_aligned(latent, self.context, checkpointed)
        self.rgb_decoder_calls += 1
        return rgb

    def commit_reference_once(self):
        if self.reference_latent is None:
            raise RuntimeError("fixed reference prefix has not arrived")
        with torch.no_grad():
            feature = self.codec.projector(self.reference_latent, self.context.intra)
        return self.context.commit_reference_once(feature, self.reference_latent)


class AIPVTokCodec(nn.Module):
    def __init__(self, backbone, projector, reference_i=2048, reference_p=510):
        super().__init__()
        self.backbone, self.projector = backbone, projector
        self.backbone.requires_grad_(False)
        self.projector.requires_grad_(False).eval()
        self.predictor = TemporalPredictor()
        self.tokenizer = ResidualTokenizer()
        self.fusion = FusionDecoder()
        self.reference_i, self.reference_p = reference_i, reference_p
        self.encode_calls = 0

    def encode_frame(self, rgb, state, frame_id):
        context = self.backbone.begin_frame(state, frame_id)
        with torch.no_grad():
            native = self.backbone.aligned_latent(rgb, context)
        prediction = self.predictor(state.latent, context.intra, native.shape[-2:], rgb.device)
        self.encode_calls += 1
        return EncodedFrame(frame_id, self.tokenizer.encode(native, prediction))

    def begin_frame(self, frame_id, tiers, state):
        return DecodeSession(self, state, frame_id, tiers)

    def prefix_session(self, encoded, state, budget, differentiable=False):
        session = self.begin_frame(encoded.frame_id, encoded.tokens.tiers, state)
        values = encoded.tokens.codes if differentiable else encoded.tokens.ids
        if budget > values.shape[1]:
            raise ValueError("requested budget exceeds encoded tokens")
        for offset in range(0, budget, 128):
            session.append(values[:, offset : min(offset + 128, budget)], offset, codes=differentiable)
        return session


def create_codec(config):
    backbone = create_backbone(config)
    load_alignment(config, backbone)
    path = resolve(config["checkpoints"]) / "projector_w128/latest.pt"
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    projector = LatentToRef(artifact["width"]).cuda()
    projector.load_state_dict(artifact["projector"])
    return AIPVTokCodec(backbone, projector).cuda().to(memory_format=torch.channels_last)
