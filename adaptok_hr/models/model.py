"""Frame-level and sequence-level APIs share the same causal transition."""

from dataclasses import dataclass

import torch
from torch import nn

from .decoder import Decoder
from .encoder import Encoder
from .temporal_state import TemporalState, frame_context, stream_identity
from .tokenizer import PositionTable, Tokenizer
from .vq import VectorQuantizer


@dataclass
class FrameOutput:
    ids: torch.Tensor
    rgb: torch.Tensor
    feature: torch.Tensor
    vq_loss: torch.Tensor
    stats: dict
    state: TemporalState
    frame_type: str


class AdapTokHR(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.positions = PositionTable(config)
        self.encoder = Encoder(config)
        self.tokenizer = Tokenizer(config, self.positions)
        self.vq = VectorQuantizer(config.codebook_size, config.embedding_dim,
                                  codebook_mode=config.codebook_mode, loss_scope=config.loss_scope,
                                  counts=self.positions.counts)
        self.decoder = Decoder(config, self.positions)

    def validate_budget(self, budget):
        if type(budget) is not int or budget not in self.config.budgets:
            raise ValueError(f"allowed budgets: {self.config.budgets}")

    def validate_frame(self, frame):
        if frame.ndim != 4 or frame.shape[1:] != (3, self.config.height, self.config.width):
            raise ValueError("frame shape must be B,3,configured height,width")
        if not frame.is_floating_point() or not torch.isfinite(frame).all():
            raise ValueError("finite floating RGB input required")
        if frame.min() < 0 or frame.max() > 1:
            raise ValueError("RGB must be in [0,1]")

    def encode_features(self, frame, state=None):
        self.validate_frame(frame)
        _, _, reference = frame_context(state, self.config.gop)
        features = self.encoder(frame, reference)
        return self.vq(self.tokenizer(features))

    def token_ids(self, frame, state, budget):
        self.validate_budget(budget)
        return self.encode_features(frame, state).ids[:, :budget]

    def decode_ids(self, ids, state=None):
        if ids.ndim != 2 or ids.dtype != torch.long:
            raise ValueError("ID input must be a B,K long tensor")
        self.validate_budget(ids.shape[1])
        if ids.min() < 0 or ids.max() >= self.config.codebook_size:
            raise ValueError("code ID out of range")
        _, _, reference = frame_context(state, self.config.gop)
        return self.decoder(self.vq.lookup(ids), reference)

    def forward_frame(self, frame, state, budget, encoded=None):
        self.validate_budget(budget)
        index, frame_type, reference = frame_context(state, self.config.gop)
        encoded = self.encode_features(frame, state) if encoded is None else encoded
        rgb, feature = self.decoder(encoded.quantized[:, :budget], reference)
        next_state = TemporalState(stream_id=stream_identity("training") if state is None else state.stream_id,
                                   frame_index=index, frame_type=frame_type, gop=self.config.gop,
                                   rgb=rgb, feature=feature)
        return FrameOutput(encoded.ids[:, :budget], rgb, feature, encoded.loss_terms(budget)["vq_loss"], encoded.stats,
                           next_state, frame_type)

    def forward(self, video, budget=2048, state=None):
        if video.ndim != 5 or video.shape[1] != 3 or video.shape[2] < 1:
            raise ValueError("video must be B,3,T,H,W")
        outputs = []
        for frame in video.unbind(2):
            output = self.forward_frame(frame, state, budget)
            outputs.append(output)
            state = output.state
        return outputs
