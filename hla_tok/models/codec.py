"""Frame-local HLA-Tok analysis and synthesis from discrete prefixes."""

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.profiler import record_function

from hla_tok.models.local import RANK_CHUNK, LocalQueryEncoder, LocalTokenDecoder, Memory
from hla_tok.models.quantizer import VectorQuantizer
from hla_tok.models.spatial import RGBDecoder, SpatialEncoder
from hla_tok.progressive.geometry import MAX_QUERIES, Geometry, group_prefix, merge_regions, rank_major


@dataclass
class SenderFrame:
    geometry: Geometry
    batch: int
    memory: tuple[Memory, ...]
    autocast_dtype: torch.dtype | None
    chunks: list[Tensor] = field(default_factory=list)


@dataclass
class TrainingTokens:
    geometry: Geometry
    ids: Tensor | None
    values: Tensor
    continuous: Tensor
    codebook_loss: Tensor
    commitment_loss: Tensor


class HLATok(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = SpatialEncoder()
        self.query_encoder = LocalQueryEncoder()
        self.quantizer = VectorQuantizer()
        self.local_decoder = LocalTokenDecoder()
        self.rgb_decoder = RGBDecoder()

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
        return SenderFrame(geometry, rgb.shape[0], memory, precision)

    @torch.no_grad()
    def prepare_frame(self, rgb: Tensor) -> SenderFrame:
        return self._prepare(rgb)

    def _block(self, sender: SenderFrame, start: int) -> Tensor:
        with record_function("hla/local_query"):
            with torch.autocast(sender.memory[0][0].device.type, dtype=sender.autocast_dtype,
                                enabled=sender.autocast_dtype is not None):
                block = self.query_encoder(sender.memory, start)
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
        return self.encode_prefix(sender, sender.geometry.capacity if budget is None else budget)

    def encode_prefix(self, sender: SenderFrame, budget: int) -> Tensor:
        return self.extend_prefix(sender, budget)

    @torch.no_grad()
    def extend_prefix(self, sender: SenderFrame, budget: int) -> Tensor:
        sender.geometry.check_budget(budget)
        block_size = sender.geometry.num_regions * RANK_CHUNK
        required = (budget + block_size - 1) // block_size
        for index in range(len(sender.chunks), required):
            sender.chunks.append(self._ordered(self._ids(self._block(sender, index * RANK_CHUNK))))
        return torch.cat(sender.chunks[:required], dim=1)[:, :budget].contiguous()

    @torch.no_grad()
    def encode_regions(self, sender: SenderFrame, lengths: tuple[int, ...]) -> tuple[Tensor, ...]:
        regions = sender.geometry.num_regions
        if len(lengths) != regions or any(type(n) is not int or not 0 <= n <= MAX_QUERIES for n in lengths):
            raise ValueError("expected a legal local prefix length for each region")
        if max(lengths) == 0:
            device = sender.memory[0][0].device
            return tuple(torch.empty(sender.batch, 0, dtype=torch.long, device=device) for _ in lengths)
        ordered = self.extend_prefix(sender, max(lengths) * regions).reshape(sender.batch, -1, regions)
        return tuple(ordered[:, :count, index] for index, count in enumerate(lengths))

    def _decode_features(self, values: Tensor, geometry: Geometry) -> Tensor:
        with record_function("hla/local_decoder"):
            local = [self.local_decoder(group) for group in group_prefix(values, geometry.num_regions)]
            return merge_regions(torch.cat(local, dim=1), geometry)

    def _decode(self, values: Tensor, geometry: Geometry) -> Tensor:
        return self.rgb_decoder(self._decode_features(values, geometry), geometry)

    @torch.no_grad()
    def decode_prefix(self, ids: Tensor, geometry: Geometry) -> Tensor:
        if ids.ndim != 2 or ids.dtype != torch.long or ids.shape[0] == 0:
            raise ValueError("expected nonempty batched int64 sequences")
        geometry.check_budget(ids.shape[1])
        with record_function("hla/codebook_lookup"):
            values = self.quantizer.lookup(ids)
        return self._decode(values, geometry)

    def encode_training(self, rgb: Tensor, budget: int, *, quantize: bool = True) -> TrainingTokens:
        sender = self._prepare(rgb)
        sender.geometry.check_budget(budget)
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
