"""Official DCVC signed rANS records and conditional precision refinements."""

from typing import Any

import numpy as np
import torch
from torch import Tensor

from pht_tok.precision import WIDTHS, PrecisionTables, refinement_context, representative


class NativeEntropy:
    def __init__(self, model: Any):
        from MLCodec_extensions_cpp import RansEncoder
        from pht_entropy_native import SignedDecoder

        self.y_tables = y, y_sizes = model.gaussian_encoder.get_cdf_info()
        self.z_tables = z, z_sizes = model.bit_estimator_z.get_cdf_info()
        self._writer_type = RansEncoder
        self._mask_writers: tuple[Any, ...] | None = None
        self.writer = RansEncoder()
        self.writer.set_entropy_coder_parallel(1)
        self.writer.set_cdf(y, y_sizes, 1)
        self.writer.set_cdf(z, z_sizes, 0)
        self.y_decoder = SignedDecoder(y, y_sizes, True)
        self.z_decoder = SignedDecoder(z, z_sizes, False)
        self._precision: ConditionalEntropy | None = None

    @property
    def precision(self) -> "ConditionalEntropy":
        if self._precision is None:
            self._precision = ConditionalEntropy(PrecisionTables.from_original(*self.y_tables))
        return self._precision

    def start_mask_encoding(self, hyper: np.ndarray, qp: int, channels: int,
                            layers: tuple[np.ndarray, ...]) -> None:
        if self._mask_writers is None:
            self._mask_writers = tuple(self._writer_type() for _ in range(4))
            for writer in self._mask_writers:
                writer.set_entropy_coder_parallel(1)
                writer.set_cdf(*self.y_tables, 1)
        self.writer.reset()
        self.writer.encode_z(hyper, qp * channels, channels)
        self.writer.flush()
        for writer, packed in zip(self._mask_writers, layers, strict=True):
            writer.reset()
            writer.encode_y(packed)
            writer.flush()

    def finish_mask_encoding(self) -> tuple[bytes, ...]:
        assert self._mask_writers is not None
        return tuple(writer.get_encoded_stream().tobytes() for writer in (self.writer, *self._mask_writers))

    def encode_z(self, hyper: np.ndarray, qp: int, channels: int) -> bytes:
        self.writer.reset()
        self.writer.encode_z(hyper, qp * channels, channels)
        self.writer.flush()
        return self.writer.get_encoded_stream().tobytes()

    def decode_y(self, payload: bytes, indexes: np.ndarray) -> np.ndarray:
        return self.y_decoder.y(payload, indexes)

    def decode_z(self, payload: bytes, qp: int, shape: tuple[int, ...], device: torch.device) -> Tensor:
        _, channels, height, width = shape
        values = self.z_decoder.z(payload, channels * height * width, qp * channels, channels)
        return torch.from_numpy(values.reshape(1, height, width, channels)).to(device).permute(0, 3, 1, 2)


class ConditionalEntropy:
    def __init__(self, tables: PrecisionTables):
        from pht_entropy_native import PrecisionCoder
        self.tables = tables
        self.coders = tuple(PrecisionCoder(table) for table in tables.levels)

    def encode(self, indexes: np.ndarray, residuals: np.ndarray) -> tuple[bytes, ...]:
        exact = residuals.astype(np.int16) + 128
        parents = exact // WIDTHS[0]
        layers = [self.coders[0].encode(parents.astype(np.uint8), indexes.astype(np.uint16))]
        for width, coder in zip(WIDTHS[1:], self.coders[1:], strict=True):
            children = exact // width
            contexts = refinement_context(indexes, parents, width)
            layers.append(coder.encode((children % 2).astype(np.uint8), contexts))
            parents = children
        return tuple(layers)

    def decode(self, payload: bytes, indexes: np.ndarray, stage: int, parents: np.ndarray | None) -> np.ndarray:
        if parents is None:
            if stage != 0:
                raise ValueError("missing coarse precision intervals")
            return self.coders[0].decode(payload, indexes.astype(np.uint16))
        width = WIDTHS[stage]
        updates = self.coders[stage].decode(payload, refinement_context(indexes, parents, width))
        return parents.astype(np.uint16) * 2 + updates

    @staticmethod
    def preview(parents: np.ndarray, stage: int) -> np.ndarray:
        width = WIDTHS[stage]
        lows = parents.astype(np.int16) * width - 128
        return representative(lows, width).astype(np.int8)
