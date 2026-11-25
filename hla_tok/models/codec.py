"""Frame-local HLA-Tok analysis and synthesis from discrete prefixes."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.profiler import record_function

from hla_tok.models.anchors import DIRECT_CAPACITY, anchor_digest
from hla_tok.models.local import (
    RANK_CHUNK,
    AnchoredQueryEncoder,
    DirectProjection,
    LocalQueryEncoder,
    LocalTokenDecoder,
    Memory,
)
from hla_tok.models.quantizer import VectorQuantizer
from hla_tok.models.rank_causal import RankHistory
from hla_tok.models.spatial import RGBDecoder, SpatialEncoder
from hla_tok.progressive.geometry import MAX_QUERIES, Geometry, group_prefix, merge_regions, rank_major
from hla_tok.streaming.fixed13 import unpack_tokens_13bit


class Architecture(str, Enum):
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"

    @property
    def local_capacity(self) -> int:
        return DIRECT_CAPACITY if self is self.R1 else MAX_QUERIES

    def query_encoder(self) -> nn.Module:
        if self is self.R0:
            return LocalQueryEncoder()
        if self is self.R1:
            return DirectProjection()
        return AnchoredQueryEncoder(causal=self is self.R3)


@dataclass
class SenderFrame:
    geometry: Geometry
    batch: int
    memory: tuple[Memory, ...] | Tensor
    autocast_dtype: torch.dtype | None
    device: torch.device
    rank_history: RankHistory | None = None
    chunks: list[Tensor] = field(default_factory=list)


@dataclass
class TrainingTokens:
    geometry: Geometry
    ids: Tensor | None
    values: Tensor
    continuous: Tensor
    codebook_loss: Tensor
    commitment_loss: Tensor


def recover_groups(decoder: LocalTokenDecoder, groups: Iterable[tuple[Tensor, Tensor | None]],
                   geometry: Geometry) -> Tensor:
    """Synthesize spatial grids from rasters of regions with equal received Rank sets."""
    with record_function("hla/local_decoder"):
        local = [decoder(group) if ranks is None else decoder(group, ranks) for group, ranks in groups]
        return merge_regions(torch.cat(local, dim=1), geometry)


def recover_grid(decoder: LocalTokenDecoder, values: Tensor, geometry: Geometry,
                 ranks: Tensor | None = None) -> Tensor:
    """Recover native spatial features from a Rank-major prefix."""
    groups = group_prefix(values, geometry.num_regions)
    return recover_groups(decoder, ((group, None if ranks is None else ranks[:group.shape[2]])
                                    for group in groups), geometry)


class PrefixDecoding:
    architecture: Architecture
    quantizer: VectorQuantizer
    local_decoder: LocalTokenDecoder
    rgb_decoder: RGBDecoder

    def _capacity(self, geometry: Geometry) -> int:
        return self.architecture.local_capacity * geometry.num_regions

    def _check_budget(self, geometry: Geometry, budget: int) -> None:
        geometry.check_budget(budget)
        if budget > self._capacity(geometry):
            raise ValueError("budget exceeds the selected architecture's spatial capacity")

    def _decode_features(self, values: Tensor, geometry: Geometry) -> Tensor:
        return recover_grid(self.local_decoder, values, geometry)

    def _decode(self, values: Tensor, geometry: Geometry) -> Tensor:
        return self.rgb_decoder(self._decode_features(values, geometry), geometry)

    @torch.no_grad()
    def decode_prefix(self, ids: Tensor, geometry: Geometry) -> Tensor:
        if ids.ndim != 2 or ids.dtype != torch.long or ids.shape[0] == 0:
            raise ValueError("expected nonempty batched int64 sequences")
        self._check_budget(geometry, ids.shape[1])
        with record_function("hla/codebook_lookup"):
            values = self.quantizer.lookup(ids)
        return self._decode(values, geometry)


class PrefixReceiver(nn.Module, PrefixDecoding):
    """Synthesis-only endpoint with no sender modules or frame state."""

    def __init__(self, architecture: Architecture, quantizer: VectorQuantizer,
                 local_decoder: LocalTokenDecoder, rgb_decoder: RGBDecoder):
        super().__init__()
        self.architecture = architecture
        self.quantizer = quantizer
        self.local_decoder = local_decoder
        self.rgb_decoder = rgb_decoder

    def decode_bytes(self, payload: bytes, count: int, geometry: Geometry) -> Tensor:
        self._check_budget(geometry, count)
        with record_function("hla/fixed13_deserialize"):
            ids = unpack_tokens_13bit(payload, count)
        with record_function("hla/host_device_ids"):
            received = torch.tensor([ids], dtype=torch.long, device=self.quantizer.embedding.weight.device)
        return self.decode_prefix(received, geometry)


class HLATok(nn.Module, PrefixDecoding):
    def __init__(self, architecture: Architecture | str = Architecture.R0, *, rgb_variant: str = "D0"):
        super().__init__()
        self.architecture = Architecture(architecture)
        self.encoder = SpatialEncoder()
        self.query_encoder = self.architecture.query_encoder()
        self.quantizer = VectorQuantizer()
        self.local_decoder = LocalTokenDecoder()
        self.rgb_decoder = RGBDecoder(rgb_variant)

    def architecture_metadata(self) -> dict:
        table = getattr(self.query_encoder, "anchor_table", None)
        metadata = {
            "architecture_version": f"hla-tok.{self.architecture.value}",
            "anchor_table_sha256": None if table is None else anchor_digest(table),
            "rank_causal": {"enabled": self.architecture is Architecture.R3, "layers": int(self.architecture is Architecture.R3),
                            "rank_chunk": RANK_CHUNK, "heads": 4, "ffn_ratio": 2, "dropout": 0.0},
        }
        return metadata if self.rgb_decoder.variant == "D0" else metadata | {"rgb_decoder": self.rgb_decoder.variant}

    def receiver(self) -> PrefixReceiver:
        return PrefixReceiver(self.architecture, self.quantizer, self.local_decoder, self.rgb_decoder)

    def _prepare(self, rgb: Tensor) -> SenderFrame:
        if rgb.ndim != 4 or rgb.shape[1] != 3 or rgb.shape[0] == 0 or not rgb.is_floating_point():
            raise ValueError("expected nonempty floating-point [batch, 3, height, width] RGB")
        geometry = Geometry(*rgb.shape[-2:])
        with record_function("hla/spatial_encoder"):
            fine, coarse = self.encoder(rgb, geometry)
        with record_function("hla/query_prepare"):
            memory = self.query_encoder.prepare(fine, coarse, geometry)
        device = rgb.device.type
        precision = torch.get_autocast_dtype(device) if torch.is_autocast_enabled(device) else None
        history = RankHistory() if self.architecture is Architecture.R3 else None
        return SenderFrame(geometry, rgb.shape[0], memory, precision, rgb.device, history)

    @torch.no_grad()
    def prepare_frame(self, rgb: Tensor) -> SenderFrame:
        return self._prepare(rgb)

    def _block(self, sender: SenderFrame, start: int) -> Tensor:
        with record_function("hla/local_query"):
            with torch.autocast(sender.device.type, dtype=sender.autocast_dtype,
                                enabled=sender.autocast_dtype is not None):
                block = self.query_encoder(sender.memory, start, sender.rank_history)
        return block.unflatten(0, (sender.batch, sender.geometry.num_regions))

    def _ids(self, z: Tensor) -> Tensor:
        with record_function("hla/vq_search"):
            return self.quantizer(z)

    @staticmethod
    def _ordered(local: Tensor) -> Tensor:
        with record_function("hla/token_ordering"):
            return rank_major(local)

    def encode_frame(self, rgb: Tensor, budget: int | None = None) -> Tensor:
        sender = self.prepare_frame(rgb)
        return self.encode_prefix(sender, self._capacity(sender.geometry) if budget is None else budget)

    def encode_prefix(self, sender: SenderFrame, budget: int) -> Tensor:
        return self.extend_prefix(sender, budget)

    @torch.no_grad()
    def extend_prefix(self, sender: SenderFrame, budget: int) -> Tensor:
        self._check_budget(sender.geometry, budget)
        block_size = sender.geometry.num_regions * RANK_CHUNK
        required = (budget + block_size - 1) // block_size
        for index in range(len(sender.chunks), required):
            sender.chunks.append(self._ordered(self._ids(self._block(sender, index * RANK_CHUNK))))
        return torch.cat(sender.chunks[:required], dim=1)[:, :budget].contiguous()

    @torch.no_grad()
    def encode_regions(self, sender: SenderFrame, lengths: tuple[int, ...]) -> tuple[Tensor, ...]:
        regions = sender.geometry.num_regions
        if len(lengths) != regions or any(type(n) is not int or not 0 <= n <= self.architecture.local_capacity for n in lengths):
            raise ValueError("expected a legal local prefix length for each region")
        if max(lengths) == 0:
            device = sender.device
            return tuple(torch.empty(sender.batch, 0, dtype=torch.long, device=device) for _ in lengths)
        ordered = self.extend_prefix(sender, max(lengths) * regions).reshape(sender.batch, -1, regions)
        return tuple(ordered[:, :count, index] for index, count in enumerate(lengths))

    def encode_training(self, rgb: Tensor, budget: int, *, quantize: bool = True) -> TrainingTokens:
        sender = self._prepare(rgb)
        self._check_budget(sender.geometry, budget)
        regions = sender.geometry.num_regions
        max_rank = (budget + regions - 1) // regions
        chunks = [self._block(sender, start) for start in range(0, max_rank, RANK_CHUNK)]
        continuous = torch.cat([self._ordered(chunk) for chunk in chunks], dim=1)[:, :budget]
        if not quantize:
            zero = continuous.new_zeros(())
            return TrainingTokens(sender.geometry, None, continuous, continuous, zero, zero)
        ids = torch.cat([self._ordered(self._ids(chunk)) for chunk in chunks], dim=1)[:, :budget]
        codes = self.quantizer.lookup(ids).float()
        z = continuous.float()
        codebook = F.mse_loss(codes, z.detach())
        commitment = F.mse_loss(z, codes.detach())
        # Hard lookup in the forward pass, identity gradient to the query output.
        values = codes.detach() + (z - z.detach())
        return TrainingTokens(sender.geometry, ids, values, continuous, codebook, commitment)

    def decode_training(self, tokens: TrainingTokens, budget: int) -> Tensor:
        if not 0 < budget <= tokens.values.shape[1]:
            raise ValueError("training decode requires a prefix of the encoded sequence")
        return self._decode(tokens.values[:, :budget], tokens.geometry)
