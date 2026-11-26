"""Plan consecutive native coding units before reading any source frames."""

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Unit:
    start: int
    valid: int
    coded: int
    intra: bool
    reset: bool

    def __post_init__(self) -> None:
        if self.start < 0 or not 1 <= self.valid <= self.coded:
            raise ValueError("invalid native frame range")
        if self.intra and (self.coded != 1 or self.reset):
            raise ValueError("an independent I unit must contain one frame")

    @property
    def indices(self) -> range:
        return range(self.start, self.start + self.valid)


@dataclass(frozen=True, slots=True)
class NativeSchedule:
    delay: int
    gop_frames: int = 0
    reset_interval: int = 0

    def __post_init__(self) -> None:
        if self.delay < 1 or self.reset_interval < 0 or self.gop_frames < 0:
            raise ValueError('invalid native coding schedule')
        if self.gop_frames and (self.gop_frames - 1) % self.delay:
            raise ValueError('a GOP must have one I frame followed by whole P units')

    def units(self, frame_count: int) -> Iterator[Unit]:
        if frame_count < 1:
            raise ValueError('a sequence needs at least one frame')
        gop = self.gop_frames or frame_count
        for first in range(0, frame_count, gop):
            yield Unit(first, 1, 1, True, False)
            stop = min(first + gop, frame_count)
            for start in range(first + 1, stop, self.delay):
                valid = min(stop - start, self.delay)
                reset = (self.reset_interval > 0
                         and (start - first + self.delay - 1) % self.reset_interval == 0)
                yield Unit(start, valid, self.delay, False, reset)


def pad_indices(unit: Unit) -> tuple[int, ...]:
    return (*unit.indices, *((unit.indices[-1],) * (unit.coded - unit.valid)))
