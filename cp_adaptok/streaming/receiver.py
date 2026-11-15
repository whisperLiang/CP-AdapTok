import time

import torch

from cp_adaptok.profiling import measure

from .bitstream import FrameHeader, StreamError
from .packetizer import ProgressivePacketStream, validate_budgets
from .sender import ids_digest, synchronize


class CodecDecoder:
    """Independent state owner. The only frame input is validated serialized bytes."""

    def __init__(self, model):
        if model.training:
            raise ValueError("stream decoder requires eval model")
        self.model = model
        self.entropy = model.entropy_model.export()
        self.codebook_id = model.vq.codebook_id()
        self.reset()

    def reset(self):
        self.state = self.model.empty_state()
        self.stream = self.context = self.cache = self.display = self.base_frame = None
        self.context_builds = self.base_commits = self.reconstructions = 0
        self.last_timing_ms = {}

    def get_state(self):
        return self.state

    def start_frame(self, header_bytes):
        h = FrameHeader.from_bytes(header_bytes)
        c = self.model.config
        expected = (
            self.model.model_id,
            self.codebook_id,
            self.entropy.entropy_id,
            c.n_max,
            c.k_ref,
            c.packet_tokens,
            c.codebook_size,
        )
        actual = (h.model_id, h.codebook_id, h.entropy_id, h.n_max, h.k_ref, h.packet_tokens, h.codebook_size)
        if expected != actual:
            raise StreamError(
                "model_id/codebook/entropy/config mismatch; explicit resynchronization required"
            )
        if self.context is not None and not self.context.committed:
            raise StreamError("previous reference prefix missing; recover or reset explicitly")
        if self.state.frame_id >= 0 and h.frame_id != self.state.frame_id + 1:
            raise StreamError("frame_id is duplicate, missing or out of order")
        fingerprint = "00" * 16 if h.frame_type == "I" else self.state.fingerprint()
        if h.reference_fingerprint != fingerprint:
            raise StreamError("reference fingerprint mismatch; explicit resynchronization required")
        try:
            self.context = self.model.begin_frame(
                self.state, h.frame_id, h.frame_type, h.height, h.width, h.qp
            )
        except ValueError as error:
            raise StreamError(str(error)) from error
        self.context_builds += 1
        self.stream = ProgressivePacketStream(h, self.entropy)
        self.cache = self.display = self.base_frame = None

    def append_packets(self, packets):
        if self.stream is None:
            raise StreamError("frame header must arrive first")
        with measure(self.model, "decoder.packet_parse_rans"):
            for packet in packets:
                self.stream.append(packet)
        return len(self.stream.token_ids)

    def _ids(self, k):
        return torch.tensor([self.stream.token_ids[:k]], dtype=torch.long, device=self.model.device)

    @torch.no_grad()
    def decode_base(self):
        if self.stream is None or len(self.stream.token_ids) < self.model.config.k_ref:
            raise StreamError("reference prefix is incomplete; reliable recovery or reset required")
        if self.context.committed:
            return self.base_frame
        ids = self._ids(self.model.config.k_ref)
        frame, feature = self.model.decode_ids(ids, self.context)
        with measure(self.model, "decoder.state_commit"):
            self.state = self.context.commit(feature, ids_digest(ids)).detach()
            self.state.fingerprint()
        self.base_commits += 1
        self.reconstructions += 1
        self.base_frame = self.display = frame
        return frame

    @torch.no_grad()
    def decode_prefix(self, k=None, use_cache=False):
        if self.stream is None:
            raise StreamError("frame header must arrive first")
        k = len(self.stream.token_ids) if k is None else k
        h = self.stream.header
        try:
            validate_budgets([k], h)
        except ValueError as error:
            raise StreamError(str(error)) from error
        if k > len(self.stream.token_ids):
            raise StreamError("display prefix must already be received")
        self.decode_base()
        if k == h.k_ref:
            self.display = self.base_frame
            self.last_timing_ms = {"attention_update_ms": 0.0, "latent_ld_reconstruction_ms": 0.0}
            return self.display
        synchronize(self.model.device)
        start = time.perf_counter()
        with measure(self.model, "decoder.prefix_attention_cache"):
            grid = self._prefix_grid(k, use_cache)
        synchronize(self.model.device)
        attention_end = time.perf_counter()
        with measure(self.model, "decoder.grid_to_latent"):
            latent = self.model.decoder.from_grid(grid, self.context.latent_shape, self.context.frame_type)
        self.display, _ = self.model.decode_latent(latent, self.context)
        self.reconstructions += 1
        synchronize(self.model.device)
        end = time.perf_counter()
        self.last_timing_ms = {
            "attention_update_ms": 1000 * (attention_end - start),
            "latent_ld_reconstruction_ms": 1000 * (end - attention_end),
        }
        return self.display

    def _prefix_grid(self, k, use_cache):
        projector = self.model.decoder.projector
        if not use_cache:
            return projector(self.model.vq.embed(self._ids(k)))
        if self.cache is None:
            self.cache = projector.new_cache(1)
        if k < self.cache.tokens:
            raise StreamError("prefix cache cannot rewind; request the reference decoder explicitly")
        if k == self.cache.tokens:
            return projector._finish(self.cache.weighted_value / self.cache.exp_sum)
        first = self.cache.tokens
        embeddings = self.model.vq.embed(self._ids(k)[:, first:])
        return projector.append(self.cache, embeddings, first)

    def get_display_frame(self):
        if self.display is None:
            raise StreamError("no displayed frame")
        return self.display
