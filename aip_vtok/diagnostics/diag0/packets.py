"""Sparse packet construction and receiver reconstruction in the fixed PCA space."""

from dataclasses import dataclass

import numpy as np
import torch

from aip_vtok.tokenizer.fsq import FSQ

from .basis import latent_grid, latent_rows
from .fit import Fit
from .sparse import (
    BUDGETS,
    GROUPS,
    Symbols,
    allocate_depth,
    byte_ledger,
    depth_bytes,
    pack_depth,
    prefix_slots,
    slot_energy,
    symbol_histogram,
    top_slots,
    unpack_depth,
)

FREE = "free_topk"
FULL = "fullres_depth_68x120"
COARSE = "coarse_depth_34x60"
HYBRID = "hybrid_anchor_plus_depth"
LAYOUTS = (FULL, COARSE, HYBRID)
VERSIONS = ("continuous", "fsq_quantized")
ANCHOR_TOKENS = 510


@dataclass(frozen=True)
class Packet:
    support: bytes | torch.Tensor
    symbols: Symbols
    anchor: Symbols | None


@dataclass(frozen=True)
class Transform:
    fit: Fit
    order: str
    version: str
    layout: str
    fsq: FSQ
    shape: tuple[int, int]

    @property
    def cell(self):
        return 2 if self.layout == COARSE else 1

    @property
    def depth_shape(self):
        return tuple(size // self.cell for size in self.shape)

    @property
    def scales(self):
        store = self.fit.residual_scales if self.layout == HYBRID else self.fit.scales
        return store[self.order]

    def anchor(self, y):
        coeffs = self.fit.block4.rotate(latent_rows(y, 4), self.fit.block4.vectors[:, :6])
        return Symbols.encode(coeffs, self.fit.anchor_scale, self.fsq, self.version)

    def decode_anchor(self, payload):
        codes = payload.decode(self.fit.anchor_scale, self.fsq)
        rows = self.fit.block4.restore(codes, self.fit.block4.vectors[:, :6])
        return latent_grid(rows, self.shape, 4)

    def positions(self, support):
        if self.layout == FREE:
            return support
        slots = prefix_slots(unpack_depth(support, self.depth_shape), self.cell)
        return torch.from_numpy(slots).to(self.scales.device)

    def supports(self, groups, budgets, anchor_tokens):
        energy = slot_energy(groups, self.fit.weights(self.order))
        if self.layout == FREE:
            ranking = top_slots(energy, max(budgets))
            return {k: ranking[:k] for k in budgets}
        # A requested subset always passes the same protocol milestones.
        milestones = {k - anchor_tokens for k in (*BUDGETS, *budgets) if k <= max(budgets)}
        depth = allocate_depth(energy.double().cpu().numpy(), milestones, self.cell)
        return {k: pack_depth(depth[k - anchor_tokens]) for k in budgets}

    def encode(self, y, budgets):
        anchor = self.anchor(y) if self.layout == HYBRID else None
        target = y if anchor is None else y - self.decode_anchor(anchor)
        coords = self.fit.basis.rotate(latent_rows(target), self.fit.axes(self.order), centered=anchor is None)
        groups = coords[:, :GROUPS * 6].reshape(*self.shape, GROUPS, 6)
        anchor_tokens = 0 if anchor is None else len(anchor.values)
        packets = {}
        for k, support in self.supports(groups, budgets, anchor_tokens).items():
            slots = self.positions(support)
            data = groups.reshape(-1, 6)[slots]
            symbols = Symbols.encode(data, self.scales[slots % GROUPS], self.fsq, self.version)
            packets[k] = Packet(support, symbols, anchor)
        return packets

    def receive(self, packet):
        slots = self.positions(packet.support)
        axes = self.fit.axes(self.order)[:, :GROUPS * 6]
        codes = packet.symbols.decode(self.scales[slots % GROUPS], self.fsq)
        groups = codes.new_zeros(np.prod(self.shape) * GROUPS, 6)
        groups.index_copy_(0, slots, codes)
        rows = self.fit.basis.restore(groups.reshape(-1, GROUPS * 6), axes, centered=packet.anchor is None)
        latent = latent_grid(rows, self.shape)
        return latent if packet.anchor is None else latent + self.decode_anchor(packet.anchor)

    def histograms(self, packet):
        slots = self.positions(packet.support)
        histogram, saturated = symbol_histogram(self.fsq, packet.symbols, slots % GROUPS, GROUPS)
        if packet.anchor is None:
            return histogram, saturated
        anchor = packet.anchor
        anchor_hist, anchor_sat = symbol_histogram(self.fsq, anchor, torch.zeros_like(anchor.values), 1)
        return np.concatenate((histogram, anchor_hist)), np.concatenate((saturated, anchor_sat))

    def ledger(self, packet, budget, pixels):
        used = len(packet.symbols.values)
        anchor = 0 if packet.anchor is None else len(packet.anchor.values)
        if self.version == "fsq_quantized":
            counts, saturated = self.histograms(packet)
            pooled = counts.sum(axis=0)
        else:
            counts = saturated = pooled = None
        packing = None if self.layout == FREE else depth_bytes(packet.support, self.depth_shape)
        support = (18 * used + 7) // 8 if self.layout == FREE else 0
        return byte_ledger(budget, used + anchor, pixels, pooled, depth=packing, support=support), counts, saturated
