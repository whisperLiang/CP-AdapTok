"""The receiver accepts bytes and its own state; the sender locally decodes the same bytes."""

import hashlib
import json

import torch

from ..models.temporal_state import TemporalState, frame_context, stream_identity
from .bitstream import Packet, parse, serialize


def tensor_digest(tensors):
    digest = hashlib.sha256()
    for name, tensor in sorted(tensors.items()):
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.digest()


class Codec:
    def __init__(self, model, amp_bf16=False):
        if model.training:
            raise ValueError("Codec requires model.eval()")
        self.model = model
        self.device = next(model.parameters()).device
        self.amp_bf16 = amp_bf16 and self.device.type == "cuda"
        self.codebook_identity = tensor_digest({"codebook": model.vq.codebook.weight})
        identity = {"config": model.config.to_dict(), "amp_bf16": self.amp_bf16}
        self.model_identity = hashlib.sha256(tensor_digest(model.state_dict()) +
                                              json.dumps(identity, sort_keys=True).encode()).digest()
        self.versions = [(tensor, tensor._version) for tensor in (*model.parameters(), *model.buffers())]

    def check_model(self):
        if self.model.training or any(tensor._version != version for tensor, version in self.versions):
            raise ValueError("codec model changed after identity verification; construct a fresh eval Codec")

    def autocast(self):
        return torch.autocast(self.device.type, dtype=torch.bfloat16, enabled=self.amp_bf16)

    @torch.inference_mode()
    def encode_frame(self, frame, state=None, budget=2048, stream_id="default"):
        self.check_model()
        if frame.shape[0] != 1:
            raise ValueError("one stream per codec state required")
        index, frame_type, _ = frame_context(state, self.model.config.gop)
        with self.autocast():
            ids = self.model.token_ids(frame, state, budget)[0].cpu().tolist()
        return self.encode_ids(ids, state, index, frame_type, stream_id)

    def encode_ids(self, ids, state, index, frame_type, stream_id):
        packet = Packet(index, frame_type == "I", self.model.config.gop,
                        self.model.config.height, self.model.config.width, tuple(ids),
                        stream_identity(stream_id) if state is None else state.stream_id,
                        self.codebook_identity, self.model_identity,
                        bytes(32) if frame_type == "I" else state.chain)
        data = serialize(packet)
        _, next_state = self.decode_frame(data, state)
        return data, next_state

    @torch.inference_mode()
    def encode_adaptive_frame(self, frame, state, scorer, maximum_tokens=2048, rate_weight=0.0,
                              stream_id="default"):
        self.check_model()
        if tuple(scorer.budgets) != self.model.config.budgets or frame.shape[0] != 1:
            raise ValueError("scorer/model budgets must match; one stream per state")
        with self.autocast():
            encoded = self.model.encode_features(frame, state)
            budget = scorer.select(encoded.quantized.mean(dim=1), maximum_tokens, rate_weight)
        index, frame_type, _ = frame_context(state, self.model.config.gop)
        return self.encode_ids(encoded.ids[0, :budget].cpu().tolist(), state, index, frame_type, stream_id)

    def validate_packet(self, packet, state):
        config = self.model.config
        if packet.codebook != self.codebook_identity or packet.model != self.model_identity:
            raise ValueError("model/codebook identity mismatch")
        if (packet.height, packet.width, packet.gop) != (config.height, config.width, config.gop):
            raise ValueError("frame shape/GOP mismatch")
        self.model.validate_budget(len(packet.ids))
        if state is not None:
            self.validate_state(packet, state)
        elif not packet.intra:
            raise ValueError("missing reference state; resynchronize at an I frame")
        if packet.intra and packet.previous != bytes(32):
            raise ValueError("I frame must reset previous reference identity")

    @staticmethod
    def validate_state(packet, state):
        if state.version != 1 or state.stream_id != packet.stream_id or state.gop != packet.gop:
            raise ValueError("state version/stream/GOP mismatch")
        if packet.frame_index != state.next_index:
            raise ValueError("missing, duplicate or out-of-order frame")
        if not packet.intra and (state.rgb is None or state.feature is None or packet.previous != state.chain):
            raise ValueError("reference state mismatch")

    @torch.inference_mode()
    def decode_frame(self, data, state=None):
        self.check_model()
        packet = parse(data)
        self.validate_packet(packet, state)
        ids = torch.tensor([packet.ids], device=self.device, dtype=torch.long)
        with self.autocast():
            # An empty receiver may synchronize at index32,64,... as well as index0.
            reference = None if packet.intra else state
            rgb, feature = self.model.decode_ids(ids, reference)
        chain = hashlib.sha256(data).digest()
        next_state = TemporalState(packet.stream_id, packet.frame_index, "I" if packet.intra else "P",
                                   packet.gop, rgb, feature, chain=chain)
        return rgb, next_state
