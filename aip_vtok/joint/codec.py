"""Shared trainable codec and a receiver that accepts verified integer prefixes."""

from dataclasses import dataclass
from time import perf_counter

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from aip_vtok.backbones.dcvc_aligned import AlignedDCVC
from aip_vtok.backbones.latent_to_ref import LatentToRef
from aip_vtok.c1_followup.io import tensor_digest
from aip_vtok.decoder.causal_state import FrameContext, ReferenceState

from .bitstream import FRAME_HEADER, FrameBytes, WireFormat
from .decoder import JointSLD
from .entropy import DigitEntropy
from .scan import AREA, HEIGHT, REF_BUDGET, SLOT_COUNT, WIDTH, SharedScan
from .tokenizer import GROUP_WIDTH, GROUPS, SENT_CHANNELS, ProgressiveFSQ
from .transform import CHANNELS, LatentTransform, frame_type


def timed(device, operation, *args):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = perf_counter()
    value = operation(*args)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return value, (perf_counter() - start) * 1000


@dataclass(frozen=True)
class Profile:
    table: torch.Tensor
    matrices: tuple[torch.Tensor, torch.Tensor]
    wire: WireFormat

    def matrix(self, intra: bool) -> torch.Tensor:
        return self.matrices[frame_type(intra)]


@dataclass(frozen=True)
class EncodedFrame:
    context: FrameContext
    aligned: torch.Tensor
    ordered_ids: torch.Tensor
    wire: FrameBytes
    latency: dict[str, float]


class JointCodec(nn.Module):
    def __init__(self, backbone: AlignedDCVC, projector: LatentToRef, orders: torch.Tensor,
                 levels: torch.Tensor, scales: torch.Tensor, scan_order: torch.Tensor, temperature: float = 0.02):
        super().__init__()
        self.backbone = backbone.eval().requires_grad_(False)
        self.projector = projector.eval().requires_grad_(False)
        self.transform = LatentTransform(orders)
        self.fsq = ProgressiveFSQ(levels, scales)
        self.scan = SharedScan(scan_order, temperature)
        self.sld = JointSLD()
        self.entropy = DigitEntropy(levels)
        self.register_buffer("slot_groups", torch.arange(GROUPS).repeat_interleave(AREA), persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self.projector.eval()
        return self

    @property
    def device(self):
        return self.slot_groups.device

    @torch.no_grad()
    def profile(self) -> Profile:
        table = self.scan.table()
        matrices = self.transform.matrix(True), self.transform.matrix(False)
        return Profile(table, matrices, WireFormat(table, self.entropy.tables(), tensor_digest(self)))

    def coefficients(self, aligned: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        if aligned.shape[1:] != (CHANNELS, HEIGHT, WIDTH):
            raise ValueError("expected aligned latent [B,128,68,120]")
        values = self.transform.project(aligned, matrix)[:, :SENT_CHANNELS]
        return values.reshape(-1, GROUPS, GROUP_WIDTH, HEIGHT, WIDTH).permute(0, 1, 3, 4, 2)

    def recover(self, values: torch.Tensor, mask: torch.Tensor, context: FrameContext, budget: int,
                matrix: torch.Tensor, checkpointed: bool = False) -> torch.Tensor:
        canvas = values.reshape(-1, GROUPS, HEIGHT, WIDTH, GROUP_WIDTH).permute(0, 1, 4, 2, 3)
        canvas = canvas.reshape(-1, SENT_CHANNELS, HEIGHT, WIDTH)
        mask = mask.reshape(-1, GROUPS, HEIGHT, WIDTH)
        inputs = (canvas, mask, context, budget, self.transform.order[frame_type(context.intra)])
        z = checkpoint(self.sld, *inputs, use_reentrant=False) if checkpointed else self.sld(*inputs)
        return self.transform.project(z, matrix.T)

    def recover_ids(self, ids: torch.Tensor, table: torch.Tensor, context: FrameContext,
                    matrix: torch.Tensor) -> torch.Tensor:
        budget = ids.shape[1]
        slots = table[:budget]
        values = self.fsq.decode_ids(ids, self.slot_groups[slots], context.intra)
        canvas = values.new_zeros((ids.shape[0], SLOT_COUNT, GROUP_WIDTH)).index_copy(1, slots, values)
        mask = values.new_zeros(SLOT_COUNT).index_fill_(0, slots, 1)
        return self.recover(canvas, mask, context, budget, matrix)

    @torch.no_grad()
    def encode_frame(self, rgb: torch.Tensor, state: ReferenceState, frame_id: int, profile: Profile) -> EncodedFrame:
        context, context_ms = timed(self.device, self.backbone.begin_frame, state, frame_id)
        aligned, encoder_ms = timed(self.device, self.backbone.aligned_latent, rgb, context)
        values, transform_ms = timed(self.device, self.coefficients, aligned, profile.matrix(context.intra))
        ids, fsq_ms = timed(self.device, self.fsq.encode_ids, values, context.intra)
        ordered_ids, scan_ms = timed(self.device, ids.reshape(1, -1).index_select, 1, profile.table)
        host, transfer_ms = timed(self.device, ordered_ids.flatten().tolist)
        wire = profile.wire.encode_frame(frame_id, host)
        latency = {"encoder_ms": encoder_ms, "tx_context_ms": context_ms, "transform_ms": transform_ms,
                   "fsq_ms": fsq_ms, "scan_ms": scan_ms, "ids_transfer_ms": transfer_ms}
        return EncodedFrame(context, aligned, ordered_ids, wire, latency)

    @torch.no_grad()
    def begin_frame(self, state: ReferenceState, header: bytes, profile: Profile):
        frame = profile.wire.parse_header(header)
        return DecodeSession(self, self.backbone.begin_frame(state, frame), profile)


class DecodeSession:
    def __init__(self, codec: JointCodec, context: FrameContext, profile: Profile):
        self.codec = codec
        self.context = context
        self.profile = profile
        self._ids: list[int] = []
        self.reference_budget = REF_BUDGET[context.intra]

    @property
    def ids(self) -> list[int]:
        return self._ids.copy()

    @property
    def received(self) -> int:
        return len(self._ids)

    def append_packet(self, packet: bytes) -> None:
        start, ids = self.profile.wire.decode_packet(self.context.frame_id, packet)
        if start != self.received:
            raise ValueError("packet offset does not follow the current prefix")
        self._ids.extend(ids)

    def decode_prefix(self, data: bytes) -> None:
        """Consume a complete byte prefix, allowing extension of the accepted prefix."""
        if self.profile.wire.parse_header(data[:FRAME_HEADER.size]) != self.context.frame_id:
            raise ValueError("prefix belongs to a different frame")
        for packet in self.profile.wire.packets(data):
            start, ids = self.profile.wire.decode_packet(self.context.frame_id, packet)
            if start == self.received:
                self._ids.extend(ids)
            elif start + len(ids) > self.received or self._ids[start:start + len(ids)] != list(ids):
                raise ValueError("prefix differs from previously accepted packets")

    @torch.no_grad()
    def latent(self, budget: int) -> torch.Tensor:
        if not 0 < budget <= self.received:
            raise ValueError("requested reconstruction exceeds the received prefix")
        ids = torch.tensor([self._ids[:budget]], device=self.codec.device, dtype=torch.long)
        return self.codec.recover_ids(ids, self.profile.table, self.context, self.profile.matrix(self.context.intra))

    @torch.no_grad()
    def render(self, budget: int | None = None) -> torch.Tensor:
        latent = self.latent(self.received if budget is None else budget)
        return self.codec.backbone.decode_aligned(latent, self.context)[0]

    @torch.no_grad()
    def commit_reference(self) -> ReferenceState:
        if self.context.committed:
            raise RuntimeError("reference already committed for this frame")
        if self.received < self.reference_budget:
            raise ValueError("missing fixed reference prefix")
        latent = self.latent(self.reference_budget)
        feature = self.codec.projector(latent, self.context.intra)
        return self.context.commit_reference_once(feature, latent)
