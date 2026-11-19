"""EncodeOnce and receiver sessions with fixed-prefix latent reference commits.

Gate 1 transports integer IDs in memory only. This API deliberately does not
claim a binary bitstream or bitrate until the Gate 2 entropy path is enabled.
"""

from dataclasses import dataclass

import torch
from torch import nn

from aip_vtok.backbones.latent_to_ref import LatentToRef
from aip_vtok.backbones.temporal_predictor import TemporalPredictor
from aip_vtok.decoder.causal_state import FrameContext
from aip_vtok.decoder.fusion_decoder import FusionDecoder
from aip_vtok.decoder.latent_canvas import LatentCanvas
from aip_vtok.evaluation.evaluate import create_backbone, load_alignment
from aip_vtok.runtime import resolve
from aip_vtok.tokenizer.residual_tokenizer import ResidualTokenizer, TokenEncoding
from aip_vtok.tokenizer.tier_map import scan_plan

MODEL_VERSION = "gate1-fix-v2"
LEGACY_MODEL_VERSION = "gate1-508a97e"


@dataclass
class EncodedFrame:
    frame_id: int
    tokens: TokenEncoding
    native_latent: torch.Tensor
    context: FrameContext
    prediction: torch.Tensor


class DecodeSession:
    def __init__(self, codec, state, frame_id, tiers):
        self.codec = codec
        self.context = codec.backbone.begin_frame(state, frame_id)
        device = codec.tokenizer.lifts[0].weight.device
        plan = scan_plan(tiers, codec.tokenizer.maximum).to(device)
        prediction = codec.base_latent(state, self.context, plan.latent_shape)
        self.canvas = LatentCanvas(codec.tokenizer, plan, prediction)
        self.reference_budget = min(
            codec.reference_i if self.context.intra else codec.reference_p, len(plan.positions)
        )
        self.reference_latent = prediction.detach().clone()
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
            reference = self.reference_latent if position < self.reference_budget else None
            if codes:
                self.canvas.append_codes(part, position, reference)
            else:
                self.canvas.append_ids(part, position, reference)

    def render_frame(self, checkpointed=False, *, with_latents=False):
        raw = self.canvas.latent()
        fused = self.codec.fusion(raw, self.canvas.masks)
        rgb, _ = self.codec.backbone.decode_aligned(fused, self.context, checkpointed)
        self.rgb_decoder_calls += 1
        return (rgb, raw, fused) if with_latents else rgb

    def commit_reference_once(self):
        if self.canvas.count < self.reference_budget:
            raise RuntimeError("fixed reference prefix has not arrived")
        with torch.no_grad():
            feature = self.codec.projector(self.reference_latent, self.context.intra)
        return self.context.commit_reference_once(feature, self.reference_latent)


class AIPVTokCodec(nn.Module):
    def __init__(
        self, backbone, projector, reference_i=2048, reference_p=510, *, use_temporal_predictor=False
    ):
        super().__init__()
        self.backbone, self.projector = backbone, projector
        self.backbone.requires_grad_(False)
        self.projector.requires_grad_(False).eval()
        self.model_version = LEGACY_MODEL_VERSION if use_temporal_predictor else MODEL_VERSION
        self.predictor = TemporalPredictor() if use_temporal_predictor else None
        self.tokenizer = ResidualTokenizer()
        self.fusion = FusionDecoder()
        self.reference_i, self.reference_p = reference_i, reference_p
        self.encode_calls = 0

    def base_latent(self, state, context, shape):
        weight = self.tokenizer.lifts[0].weight
        if self.predictor is None:
            return weight.new_zeros(1, 128, *shape)
        return self.predictor(state.latent, context.intra, shape, weight.device)

    def encode_frame(self, rgb, state, frame_id):
        context = self.backbone.begin_frame(state, frame_id)
        with torch.no_grad():
            native = self.backbone.aligned_latent(rgb, context).detach()
        prediction = self.base_latent(state, context, native.shape[-2:])
        self.encode_calls += 1
        tokens = self.tokenizer.encode(
            native, prediction, measure_saturation=self.model_version == MODEL_VERSION
        )
        return EncodedFrame(frame_id, tokens, native, context, prediction)

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


def create_codec(config, *, model_version=MODEL_VERSION):
    if model_version not in (MODEL_VERSION, LEGACY_MODEL_VERSION):
        raise ValueError(f"unknown codec model version: {model_version}")
    if config.get("model_version", model_version) != model_version:
        raise ValueError("requested codec version differs from the model config")
    backbone = create_backbone(config)
    load_alignment(config, backbone)
    path = resolve(config["checkpoints"]) / "projector_w128/latest.pt"
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    projector = LatentToRef(artifact["width"]).cuda()
    projector.load_state_dict(artifact["projector"])
    codec = AIPVTokCodec(backbone, projector, use_temporal_predictor=model_version == LEGACY_MODEL_VERSION)
    return codec.cuda().to(memory_format=torch.channels_last)


def create_legacy_codec(config):
    return create_codec(config, model_version=LEGACY_MODEL_VERSION)
