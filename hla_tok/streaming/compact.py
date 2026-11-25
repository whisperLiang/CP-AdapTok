"""Stateless Compact sender and a synthesis path that consumes only received bytes."""

from collections.abc import Iterable, Iterator
from hashlib import sha256

import torch
from torch import Tensor

from hla_tok import HLATok
from hla_tok.compression.packing import pack_bf16, pack_signed, unpack_bf16, unpack_signed
from hla_tok.compression.profiles import Profile
from hla_tok.compression.scalar import StaticScalar
from hla_tok.compression.transform import CompactTransform
from hla_tok.streaming.compact_wire import CompactFrame, WireIdentity, deserialize
from hla_tok.streaming.continuous import (
    MAX_K,
    REGIONS,
    ContinuousReceiver,
    Order,
    RankOrder,
    check_budget,
    encode_continuous,
)


def native_ranks(order: RankOrder, device: str | torch.device = "cpu") -> Tensor:
    return torch.tensor([rank for rank in order.ranks for _ in range(min(REGIONS, MAX_K - rank * REGIONS))],
                        dtype=torch.long, device=device)


class CompactCodec:
    def __init__(self, profile: Profile, transform: CompactTransform, scalar: StaticScalar | None,
                 order: RankOrder, asset_sha256: str, model_sha256: str, device: str | torch.device):
        if profile.dimension != transform.dimension or profile.quantized != (scalar is not None):
            raise ValueError("transform or scalar does not match registered Profile")
        if scalar is not None and (scalar.bits != profile.bits or
                                   tuple(scalar.scales.shape) != (transform.rank_count, profile.dimension)):
            raise ValueError("scalar width or Rank table does not match registered Profile")
        # Asset and synthesis identity are bound together with this exact wire-to-native permutation.
        asset_digest, model_digest = bytes.fromhex(asset_sha256), bytes.fromhex(model_sha256)
        if len(asset_digest) != 32 or len(model_digest) != 32:
            raise ValueError("Compact assets and model require SHA-256 digests")
        binding = asset_digest + model_digest
        signature = sha256(bytes([profile.code, order.name, *order.ranks]) + binding).digest()
        self.identity = WireIdentity(profile, order.name, signature)
        self.transform = transform
        self.scalar = scalar
        self.order = order
        self.device = torch.device(device)
        self.ranks = native_ranks(order, self.device)

    @property
    def profile(self) -> Profile:
        return self.identity.profile

    def encode_codes(self, u: Tensor) -> Tensor:
        return u.to(torch.bfloat16) if self.scalar is None else self.scalar.encode(u, self.ranks[:u.shape[1]])

    def encode_tokens(self, full: Tensor, budget: int) -> Tensor:
        check_budget(budget)
        # Fixed projection work also fixes arithmetic when a kernel is sensitive to the input shape.
        projected = self.transform.encode(self.order.apply(full), self.ranks)
        return self.encode_codes(projected[:, :budget])

    def pack(self, cpu: Tensor) -> bytes:
        if cpu.ndim != 3 or cpu.shape[0] != 1 or cpu.shape[2] != self.profile.dimension:
            raise ValueError("expected codes for one Compact frame")
        return pack_bf16(cpu[0]) if self.scalar is None else pack_signed(cpu[0], self.profile.bits)

    def unpack(self, payload: bytes | memoryview) -> Tensor:
        dim = self.profile.dimension
        codes = unpack_bf16(payload, dim) if self.scalar is None else unpack_signed(payload, self.profile.bits, dim)
        check_budget(codes.shape[0])
        return codes[None]

    def dequantize(self, codes: Tensor) -> Tensor:
        return codes if self.scalar is None else self.scalar.decode(codes, self.ranks[:codes.shape[1]])

    def lift(self, u: Tensor) -> Tensor:
        return self.transform.decode(u, self.ranks[:u.shape[1]])

    def frame(self, code: Tensor, frame_id: int) -> CompactFrame:
        return CompactFrame(self.identity, frame_id, self.pack(code.cpu()))

    @torch.inference_mode()
    def encode(self, model: HLATok, rgb: Tensor, budget: int, frame_id: int) -> CompactFrame:
        check_budget(budget)
        native_budget = budget if self.order.name == Order.O0 and self.transform.projection is None else MAX_K
        tokens = encode_continuous(model, rgb, native_budget)
        if native_budget != MAX_K:
            return self.frame(self.encode_codes(tokens), frame_id)
        return self.frame(self.encode_tokens(tokens, budget), frame_id)

    def prefix_codes(self, raw: bytes) -> Tensor:
        packet = deserialize(raw)
        if packet.identity != self.identity or packet.offset:
            raise ValueError("expected a base prefix of the configured Compact identity")
        return self.unpack(packet.payload)

    @torch.inference_mode()
    def decode_payload(self, receiver: ContinuousReceiver, payload: bytes | memoryview) -> Tensor:
        codes = self.unpack(payload).to(self.device)
        return receiver.decode(self.lift(self.dequantize(codes)), self.order)

    @torch.inference_mode()
    def decode(self, receiver: ContinuousReceiver, raw: bytes) -> Tensor:
        codes = self.prefix_codes(raw).to(self.device)
        return receiver.decode(self.lift(self.dequantize(codes)), self.order)


def encode_frames(model: HLATok, codec: CompactCodec, frames: Iterable[tuple[Tensor, int]]) -> Iterator[CompactFrame]:
    """Yield each current frame at its selected budget while the input iterator advances."""
    for frame_id, (source, budget) in enumerate(frames):
        yield codec.encode(model, source, budget, frame_id)
