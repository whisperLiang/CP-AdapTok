"""Diagnostic code lengths from integer CDFs, excluding wire headers and flush."""

import numpy as np

from pht_tok.bitstream import Mode
from pht_tok.codec import Layout, flat_nhwc
from pht_tok.entropy import NativeEntropy
from pht_tok.official_adapter import EncodedLatents
from pht_tok.precision import CDF_TOTAL, WIDTHS, refinement_context


def signed_bits(cdf, lengths, indexes, symbols):
    signed = symbols.astype(np.int32)
    ordered = np.abs(signed) * 2 - (signed > 0)
    escape = lengths[indexes] - 2
    token = np.minimum(ordered, escape)
    frequency = cdf[indexes, token + 1] - cdf[indexes, token]
    values = np.maximum(ordered - escape, 0)
    digits = (np.frexp(values)[1] + 1) // 2
    bypass = np.where(ordered >= escape, 2 * (digits + digits // 3 + 1), 0)
    return float((-np.log2(frequency / CDF_TOTAL) + bypass).sum())


def precision_bits(tables, indexes, exact):
    rows = indexes
    parents = exact // WIDTHS[0]
    bits = [interval_bits(tables[0], rows, parents)]
    for width, cdf in zip(WIDTHS[1:], tables[1:], strict=True):
        rows = refinement_context(indexes, parents, width)
        child = exact // width
        bits.append(interval_bits(cdf, rows, child % 2))
        parents = child
    return bits


def interval_bits(cdf, rows, values):
    freq = cdf[rows, values + 1] - cdf[rows, values]
    return float(-np.log2(freq / CDF_TOTAL).sum())


def estimate(entropy: NativeEntropy, analysis: EncodedLatents, layout: Layout, qp: int, mode: Mode):
    z = flat_nhwc(analysis.hyper).cpu().numpy()
    channels = analysis.hyper.shape[1]
    z_cdf, z_lengths = entropy.z_tables
    hyper = signed_bits(z_cdf, z_lengths, qp * channels + np.arange(len(z)) % channels, z)
    packed = analysis.packed.cpu().numpy()
    keep = analysis.keep.cpu().numpy()
    if mode == Mode.MASK:
        selected = [packed[ids] for ids in layout.active(keep)]
        y_cdf, y_lengths = entropy.y_tables
        layers = [signed_bits(y_cdf, y_lengths, (group & 255).astype(np.uint8), group >> 8)
                  for group in selected]
    else:
        active = packed[keep]
        layers = precision_bits(entropy.precision.tables.levels, (active & 255).astype(np.uint8),
                                (active >> 8) + 128)
    return [hyper, *layers]
