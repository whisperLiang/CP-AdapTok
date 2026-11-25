"""Frame-level contract shared by native and progressive endpoints."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class FrameInfo:
    index: int
    height: int
    width: int
    qp: int
    intra: bool
    parallel: int
    reset: bool

    def __post_init__(self) -> None:
        if not 0 <= self.index < 2**32 or not (0 < self.height <= 8192 and 0 < self.width <= 8192):
            raise ValueError("invalid frame index or dimensions")
        if not 0 <= self.qp < 64 or not 1 <= self.parallel <= 8 or (self.intra and self.reset):
            raise ValueError("invalid native frame flags or QP")

    @property
    def dimensions(self) -> tuple[int, int]:
        return self.height, self.width

    @property
    def native_padding(self) -> tuple[int, int]:
        return (-self.height) % 64, (-self.width) % 64


@dataclass(frozen=True, slots=True)
class BaseFrame:
    info: FrameInfo
    payload: bytes


def schedule(index: int, intra_period: int = 32, reset_interval: int = 32) -> tuple[bool, bool]:
    intra = index == 0 or (intra_period > 0 and index % intra_period == 0)
    reset = not intra and reset_interval > 0 and index % reset_interval == 0
    return intra, reset


class ReferenceOrder:
    def __init__(self) -> None:
        self.last: FrameInfo | None = None
        self.desync = True

    def check(self, info: FrameInfo) -> None:
        prior = self.last
        if prior is not None and info.index <= prior.index:
            raise ValueError("duplicate or stale base frame")
        if info.intra:
            return
        if self.desync or prior is None or (info.index, info.dimensions) != (prior.index + 1, prior.dimensions):
            self.desync = True
            raise ValueError("DESYNC: a complete independent I-frame is required")

    def commit(self, info: FrameInfo) -> None:
        self.check(info)
        self.last = info
        self.desync = False
