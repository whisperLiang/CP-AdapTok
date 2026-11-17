"""Full-token reconstruction with explicit, independently owned causal references."""

import time
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from cp_adaptok.backbones.causal_state import CausalState, FrozenFrameContext
from cp_adaptok.backbones.intra_adapter import prepare_rgb, restore_rgb

PATHS = ('R0', 'R0-spatial', 'R1', 'R2', 'raw_192')
REFERENCES = ('i_only', 'full2048', 'original128')


@dataclass
class FrameEncoding:
    context: FrozenFrameContext
    latent: torch.Tensor
    features: torch.Tensor | None
    timings: dict


@dataclass
class Reconstruction:
    rgb: torch.Tensor
    feature: torch.Tensor
    latent: torch.Tensor
    embeddings: torch.Tensor | None
    ids: torch.Tensor | None
    vq_loss: torch.Tensor
    timings: dict


class Fixed2048Reconstructor(nn.Module):
    """Only the five reconstruction components are registered; no protocol modules.

    R1 always calls the 16D continuous projection explicitly, including in eval.
    original128 is an evaluation-only historical VQ reference, independent of the
    displayed full-token reconstruction. No candidate commits inside compare().
    """

    def __init__(self, source, identity, profile=False):
        super().__init__()
        self.config, self.identity, self.profile = source.config, identity, profile
        if (self.config.n_max, self.config.dim, self.config.embedding_dim) != (2048, 192, 16):
            raise ValueError('experiment requires 2048 tokens, 192D features and 16D projection')
        self.intra, self.ld = source.intra, source.ld
        self.tokenizer, self.vq, self.decoder = source.tokenizer, source.vq, source.decoder
        self.intra.requires_grad_(False).eval()
        self.ld.requires_grad_(False).eval()
        self.vq.set_training_policy('deterministic_nearest')
        self.training_path = 'R1'

    @property
    def device(self):
        return self.vq.project_in.weight.device

    def train(self, mode=True):
        super().train(mode)
        self.intra.eval()
        self.ld.eval()
        return self

    def configure_training(self, path):
        if path not in ('R1', 'R2', 'raw_192'):
            raise ValueError('only full-token learned paths can be trained')
        self.requires_grad_(False)
        self.tokenizer.requires_grad_(True)
        self.decoder.requires_grad_(True)
        if path != 'raw_192':
            self.vq.project_in.requires_grad_(True)
            self.vq.project_out.requires_grad_(True)
        self.vq.codebook.requires_grad_(path == 'R2')
        self.vq.set_training_policy('deterministic_nearest')
        self.training_path = path
        return self

    def empty_state(self):
        return CausalState.empty(self.identity)

    def _timed(self, name, operation, timings):
        if not self.profile:
            return operation()
        self._sync()
        start = time.perf_counter()
        output = operation()
        self._sync()
        timings[name] = (time.perf_counter() - start) * 1000
        return output

    def _sync(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)

    def begin_frame(self, rgb, state, frame_id, force_intra):
        if state.model_id != self.identity or rgb.ndim != 4 or rgb.shape[:2] != (1, 3):
            raise ValueError('expected independent B=1 RGB and matching reference identity')
        h, w = rgb.shape[-2:]
        if state.feature is None or force_intra:
            return FrozenFrameContext(self.identity, frame_id, 'I', h, w, self.config.qp, -1, '00' * 16)
        if frame_id != state.frame_id + 1:
            raise ValueError('nonconsecutive P frame reference')
        context, memory = self.ld.begin_frame(state)
        return FrozenFrameContext(self.identity, frame_id, 'P', h, w, self.config.qp,
                                  state.frame_id, state.fingerprint(), context, memory)

    def encode(self, rgb, state, frame_id, force_intra=False):
        timings = {}
        context = self._timed('reference_context_ms',
                              lambda: self.begin_frame(rgb, state, frame_id, force_intra), timings)
        adapter = self.intra if context.frame_type == 'I' else self.ld
        latent = self._timed('encoder_ms', lambda: adapter.encode_latent(prepare_rgb(rgb), context), timings)
        return FrameEncoding(context, latent, None, timings)

    def tokens(self, encoded):
        if encoded.features is None:
            encoded.features = self._timed('tokenizer_ms', lambda: self.tokenizer(
                encoded.latent, encoded.context.context, encoded.context.frame_type), encoded.timings)
            if encoded.features.shape != (1, 2048, 192):
                raise ValueError('Tokenizer must produce exactly [1,2048,192]')
        return encoded.features

    def continuous(self, features):
        with torch.autocast(device_type=features.device.type, enabled=False):
            return self.vq.project_out(self.vq.project_features(features))

    def _embeddings(self, features, path):
        if path == 'R1':
            return self.continuous(features), None, features.new_zeros(())
        if path == 'raw_192':
            return features, None, features.new_zeros(())
        result = self.vq(features)
        return result.embeddings, result.ids, result.loss

    def decode_latent(self, latent, context):
        adapter = self.intra if context.frame_type == 'I' else self.ld
        yuv, feature = adapter.decode_latent(latent, context)
        return restore_rgb(yuv, context.height, context.width), feature

    def reconstruct(self, encoded, path):
        if path not in PATHS:
            raise ValueError(f'unknown reconstruction path: {path}')
        timings, embeddings, ids = {}, None, None
        loss = encoded.latent.new_zeros(())
        latent = encoded.latent
        if path == 'R0-spatial':
            low = F.adaptive_avg_pool2d(latent, (17, 30))
            latent = F.interpolate(low, size=latent.shape[-2:], mode='bilinear', align_corners=False)
        elif path in ('R1', 'R2', 'raw_192'):
            features = self.tokens(encoded)
            embeddings, ids, loss = self._timed('vq_ms', lambda: self._embeddings(features, path), timings)
            latent = self._timed('token_to_latent_ms', lambda: self.decoder(
                embeddings, encoded.context.latent_shape, encoded.context.frame_type), timings)
        rgb, feature = self._timed('dcvc_decoder_ms', lambda: self.decode_latent(latent, encoded.context), timings)
        return Reconstruction(rgb, feature, latent, embeddings, ids, loss, timings)

    def compare(self, encoded, paths=('R0', 'R0-spatial', 'R1', 'R2')):
        if encoded.context.committed:
            raise RuntimeError('paired candidates require an uncommitted FrozenFrameContext')
        return {path: self.reconstruct(encoded, path) for path in paths}

    def advance(self, encoded, output, reference):
        if reference not in REFERENCES:
            raise ValueError('unknown reference policy')
        if reference == 'i_only':
            return self.empty_state()
        if reference == 'original128':
            if self.training:
                raise ValueError('original128 is an evaluation-only diagnostic')
            historical = self.vq(self.tokens(encoded)).embeddings[:, :128]
            latent = self.decoder(historical, encoded.context.latent_shape, encoded.context.frame_type)
            _, feature = self.decode_latent(latent, encoded.context)
        else:
            feature = output.feature
        return encoded.context.commit(feature, f'{reference}:{encoded.context.frame_id}')
