from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

class Layout:
    def __init__(self, channels: int, height: int, width: int, tile: int, band: int) -> None: ...
    @property
    def counts(self) -> NDArray[np.int32]: ...
    def gather(self, symbols: NDArray[np.int8]) -> tuple[NDArray[np.int8], NDArray[np.int32]]: ...

class Records:
    def __init__(self, tables: Sequence[Sequence[int]]) -> None: ...
    def pack(
        self, layout: Layout, values: NDArray[np.int8], scores: NDArray[np.float64],
        active: NDArray[np.uint8], cdfs: NDArray[np.uint8], ranked: bool, cap: int, fallback: bool,
    ) -> tuple[bytes, list[int]]: ...
    def receive(
        self, layout: Layout, buffer: bytearray, canvas: NDArray[np.int16], levels: NDArray[np.uint8],
        bands: int, raw: bool,
    ) -> tuple[int, int, int]: ...
