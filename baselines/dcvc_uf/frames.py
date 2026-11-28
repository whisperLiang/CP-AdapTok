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

@dataclass(frozen=True, slots=True)
class BaseFrame:
    info: FrameInfo
    payload: bytes
