"""32D frozen Rank-aware affine transform and synthesis from actual native support."""

import numpy as np
import torch
from torch import Tensor, nn

from hla_tok import HLATok
from hla_tok.models.codec import recover_grid, recover_groups
from hla_tok.models.local import LocalTokenDecoder
from hla_tok.models.spatial import RGBDecoder
from hla_tok.streaming.continuous import NATIVE
from hla_tok.temporal.layout import LAYOUT
from hla_tok.temporal.profile import RANK_COMPONENT_SHAPE
from hla_tok.temporal.state import DisplayReconstructionState


class NativeTransform(nn.Module):
    def __init__(self, center: Tensor, basis: Tensor):
        super().__init__()
        if center.shape != RANK_COMPONENT_SHAPE or basis.shape != (*RANK_COMPONENT_SHAPE, LAYOUT.components):
            raise ValueError("expected the full 32D native Rank-aware PCA")
        self.register_buffer("center", center.float().contiguous())
        self.register_buffer("basis", basis.float().contiguous())
        self._full = LAYOUT.token_count - LAYOUT.tokens_for_rank(LAYOUT.mask.shape[0] - 1).size

    def _convert(self, values: Tensor, inverse: bool) -> Tensor:
        if values.ndim != 3 or tuple(values.shape[1:]) != (LAYOUT.token_count, LAYOUT.components):
            raise ValueError("transform requires a complete native Rank-major vector sequence")
        batch, _, dim = values.shape
        ranks, regions = LAYOUT.mask.shape
        weights = self.basis.transpose(1, 2) if inverse else self.basis
        center = self.center[:, None, None]
        with torch.autocast(values.device.type, enabled=False):
            body = values[:, :self._full].float().reshape(batch, ranks - 1, regions, dim).transpose(0, 1)
            tail = values[:, self._full:].float()
            if not inverse:
                body = body - center[:-1]
                tail = tail - center[-1, 0]
            body = torch.bmm(body.reshape(ranks - 1, batch * regions, dim), weights[:-1])
            body = body.reshape(ranks - 1, batch, regions, dim)
            tail = tail @ weights[-1]
            if inverse:
                body = body + center[:-1]
                tail = tail + center[-1, 0]
            return torch.cat((body.transpose(0, 1).reshape(batch, self._full, dim), tail), dim=1)

    def project(self, values: Tensor) -> Tensor:
        return self._convert(values, inverse=False)

    def lift(self, coefficients: Tensor) -> Tensor:
        return self._convert(coefficients, inverse=True)


class TemporalSynthesis(nn.Module):
    def __init__(self, local_decoder: LocalTokenDecoder, rgb_decoder: RGBDecoder,
                 transform: NativeTransform, scale: Tensor):
        super().__init__()
        if scale.shape != RANK_COMPONENT_SHAPE or not (torch.isfinite(scale) & (scale > 0)).all():
            raise ValueError("invalid shared lattice scale")
        self.local_decoder = local_decoder
        self.rgb_decoder = rgb_decoder
        self.transform = transform
        self.register_buffer("native_scale", scale.index_select(0, torch.tensor(LAYOUT.ranks, device=scale.device)))

    @classmethod
    def from_model(cls, model: HLATok, transform: NativeTransform, scale: Tensor) -> "TemporalSynthesis":
        return cls(model.local_decoder, model.rgb_decoder, transform, scale).eval()

    def receive_coordinates(self, state: DisplayReconstructionState) -> Tensor:
        return torch.from_numpy(state.coordinates).to(self.native_scale.device)[None].float() * self.native_scale

    def recover(self, coefficients: Tensor, support: np.ndarray) -> Tensor:
        native = self.transform.lift(coefficients)
        if support.all():
            return recover_grid(self.local_decoder, native, NATIVE)
        return recover_groups(self.local_decoder, LAYOUT.recovery_groups(native, support.tobytes()), NATIVE)

    def reconstruct(self, features: Tensor) -> Tensor:
        return self.rgb_decoder(features, NATIVE)

    def render_coefficients(self, coefficients: Tensor, support: np.ndarray) -> Tensor:
        return self.reconstruct(self.recover(coefficients, support))

    @torch.inference_mode()
    def render(self, state: DisplayReconstructionState) -> Tensor:
        if state.support.shape != (LAYOUT.token_count,) or state.support.dtype != np.bool_:
            raise ValueError("display support must address the native layout")
        return self.render_coefficients(self.receive_coordinates(state), state.support)
