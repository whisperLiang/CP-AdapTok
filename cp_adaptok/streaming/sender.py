import hashlib
import time
from dataclasses import dataclass

import torch

from .bitstream import FrameHeader
from .packetizer import ProgressivePacketStream, packetize, validate_budgets


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def ids_digest(ids):
    values = ids.detach().cpu().numpy().astype(">u2", copy=False).tobytes()
    return hashlib.sha256(values).hexdigest()[:32]


@dataclass(frozen=True)
class EncodedPacketFrame:
    header: FrameHeader
    packets: tuple
    token_ids: tuple
    timing_ms: dict
    predicted_mse: dict

    @property
    def header_bytes(self):
        return self.header.to_bytes()

    def prefix_packets(self, k):
        if k < self.header.k_ref or k > self.header.n_max or k % self.header.packet_tokens:
            raise ValueError("budget must be a valid packet-aligned prefix >= K_ref")
        return self.packets[: k // self.header.packet_tokens]

    def bytes_at(self, k, transport=True):
        packets = self.prefix_packets(k)
        framing = 4 * (len(packets) + 1) if transport else 0
        return len(self.header_bytes) + sum(map(len, packets)) + framing

    def allocation_points(self, delivered_tokens=0, control_bytes=0):
        from .online_allocator import RDPoint

        if not isinstance(control_bytes, int) or control_bytes < 0:
            raise ValueError("invalid control overhead")
        if delivered_tokens:
            self.prefix_packets(delivered_tokens)
            if delivered_tokens not in self.predicted_mse:
                raise ValueError("delivered prefix must have a causal quality prediction")
        delivered_bytes = self.bytes_at(delivered_tokens) if delivered_tokens else 0
        return [
            RDPoint(
                k, self.bytes_at(k) - delivered_bytes + (control_bytes if k > delivered_tokens else 0), mse
            )
            for k, mse in self.predicted_mse.items()
            if k >= delivered_tokens
        ]


class CodecEncoder:
    def __init__(self, model, intra_period=32):
        if model.training or intra_period < 1:
            raise ValueError("stream encoder requires an eval model and positive intra period")
        self.model, self.intra_period = model, intra_period
        self.entropy = model.entropy_model.export()
        self.codebook_id = model.vq.codebook_id()
        self.reset()

    def reset(self):
        self.state = self.model.empty_state()
        self.base_commits = 0

    def get_state(self):
        return self.state

    def select_prefix(
        self,
        encoded,
        bandwidth_bps,
        deadline_s,
        queued_bytes=0,
        buffer_s=0,
        mode="scorer",
        fixed_k=None,
        delivered_tokens=0,
        control_bytes=0,
    ):
        """Select immutable packets; RDPoint.bytes charges only the remaining wire bytes."""
        from .online_allocator import OnlineTokenAllocator

        points = encoded.allocation_points(delivered_tokens, control_bytes)
        allocator = OnlineTokenAllocator(self.model.config.k_ref)
        point = allocator.allocate(points, bandwidth_bps, deadline_s, queued_bytes, buffer_s, mode, fixed_k)
        return None if point is None else (point, encoded.prefix_packets(point.tokens))

    @torch.no_grad()
    def encode_frame(self, rgb, force_intra=False, budgets=None):
        synchronize(self.model.device)
        start = time.perf_counter()
        c = self.model.config
        budgets = sorted({c.k_ref, c.n_max}) if budgets is None else budgets
        validate_budgets(budgets, c)
        frame_id = self.state.frame_id + 1
        encoded = self.model.encode_full(
            rgb, self.state, frame_id, force_intra or frame_id % self.intra_period == 0
        )
        synchronize(self.model.device)
        network_end = time.perf_counter()
        ctx = encoded.context
        header = FrameHeader(
            frame_id,
            ctx.frame_type,
            ctx.reference_id,
            ctx.height,
            ctx.width,
            c.n_max,
            c.k_ref,
            c.packet_tokens,
            c.codebook_size,
            self.model.model_id,
            self.codebook_id,
            self.entropy.entropy_id,
            ctx.reference_fingerprint,
            ctx.qp,
        )
        ids = encoded.vq.ids[0].cpu().tolist()
        packets = packetize(header, ids, self.entropy)
        entropy_end = time.perf_counter()
        # Sender reference also traverses serialized rANS packets, with no continuous side channel.
        local_stream = ProgressivePacketStream(header, self.entropy)
        for packet in packets[: c.k_ref // c.packet_tokens]:
            local_stream.append(packet)
        base_ids = torch.tensor([local_stream.token_ids], device=self.model.device, dtype=torch.long)
        _, feature = self.model.decode_ids(base_ids, ctx)
        self.state = ctx.commit(feature, ids_digest(base_ids)).detach()
        self.state.fingerprint()  # paid once in the measured state update, then safely cached
        self.base_commits += 1
        scores = self.model.scorer(encoded.features, budgets)[0].cpu().tolist()
        synchronize(self.model.device)
        end = time.perf_counter()
        timing = {
            "encode_full_ms": (network_end - start) * 1000,
            "entropy_encode_packetize_ms": (entropy_end - network_end) * 1000,
            "sender_base_decode_state_scorer_ms": (end - entropy_end) * 1000,
            "encode_total_ms": (end - start) * 1000,
        }
        return EncodedPacketFrame(header, packets, tuple(ids), timing, dict(zip(budgets, scores)))
