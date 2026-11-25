"""Static scan, quantizer, and refresh parameters bound to the HLT header."""

from dataclasses import dataclass, field, replace
from hashlib import sha256
from json import dumps

import numpy as np

from hla_tok.streaming.temporal_wire import HEADER_BYTES, RECORD_BYTES, Mode
from hla_tok.temporal.layout import LAYOUT

RANK_COMPONENT_SHAPE = (LAYOUT.mask.shape[0], LAYOUT.components)


def _immutable(array: np.ndarray, dtype: str) -> np.ndarray:
    return np.frombuffer(array.astype(dtype, copy=False).tobytes(), dtype=dtype).reshape(array.shape)


def _steps(array: np.ndarray) -> np.ndarray:
    if array.shape != RANK_COMPONENT_SHAPE or array.dtype.kind not in "iu":
        raise ValueError("expected integer steps per native rank and component")
    if array.min() < 1 or array.max() > 32767:
        raise ValueError("steps must be positive int16 multiples of the lattice")
    return _immutable(array, "<i4")


def _scan_indices(scan: np.ndarray) -> np.ndarray:
    if scan.ndim != 1 or scan.dtype.kind not in "iu" or not scan.size:
        raise ValueError("expected a nonempty scan of native component indices")
    if scan.min() < 0 or scan.max() >= LAYOUT.token_count * LAYOUT.components:
        raise ValueError("scan contains an out-of-range component")
    if np.unique(scan).size != scan.size:
        raise ValueError("scan contains overlapping components")
    return _immutable(scan, "<i4")


@dataclass(frozen=True, eq=False)
class Profile:
    profile_id: int
    scan: np.ndarray
    base_components: int
    intra_steps: np.ndarray
    inter_steps: np.ndarray
    asset_binding: bytes
    bits: int = 4
    block_size: int = 256
    sparse: bool = True
    predictor: str = "copy"
    refresh: int = 32
    scene_threshold: float | None = None
    linear_bound_steps: int = 2
    base_cap: int | None = None
    binding: bytes = field(init=False, repr=False)
    starts: tuple[int, ...] = field(init=False, repr=False)
    scan_tokens: np.ndarray = field(init=False, repr=False)
    native_base_order: np.ndarray = field(init=False, repr=False)
    scan_base_order: np.ndarray = field(init=False, repr=False)
    steps_i: np.ndarray = field(init=False, repr=False)
    steps_p: np.ndarray = field(init=False, repr=False)
    base_support: bytes = field(init=False, repr=False)
    base_records: int = field(init=False)
    max_base_bytes: int = field(init=False)

    def _check_parameters(self) -> None:
        if not 0 < self.profile_id <= 65535 or len(self.asset_binding) != 32:
            raise ValueError("invalid Profile ID or asset binding")
        if self.bits not in (4, 8) or self.block_size not in (64, 128, 256):
            raise ValueError("unsupported quantizer or short-block width")
        if self.predictor not in ("absolute", "copy", "linear") or self.refresh < 1:
            raise ValueError("invalid temporal predictor or refresh interval")
        if self.scene_threshold is not None and (not np.isfinite(self.scene_threshold) or self.scene_threshold <= 0):
            raise ValueError("scene threshold must be positive and finite")
        if not 1 <= self.linear_bound_steps <= 16:
            raise ValueError("linear prediction bound must be between 1 and 16 steps")

    def _identity(self) -> bytes:
        knobs = (self.profile_id, self.base_components, self.bits, self.block_size, self.sparse, self.predictor,
                 self.refresh, self.scene_threshold, self.linear_bound_steps, self.base_cap)
        digest = sha256(b"HLT-v1 profile\0" + self.asset_binding + dumps(knobs, separators=(",", ":")).encode())
        for array in (self.scan, self.intra_steps, self.inter_steps):
            digest.update(array.tobytes())
        return digest.digest()

    def __post_init__(self) -> None:
        self._check_parameters()
        scan = _scan_indices(self.scan)
        base, width = self.base_components, self.block_size
        if not 1 <= base <= len(scan):
            raise ValueError("invalid base scan boundary")
        steps_i, steps_p = _steps(self.intra_steps), _steps(self.inter_steps)
        tokens, components = np.divmod(scan, LAYOUT.components)
        rank, region = LAYOUT.coordinates(tokens)
        base_order = np.argsort(scan[:base])
        starts_base = tuple(range(0, base, width))
        starts = (*starts_base, *range(base, len(scan), width), len(scan))
        base_support = np.zeros(LAYOUT.token_count, dtype=bool)
        base_support[tokens[:base]] = True
        if np.unique(region[:base]).size != LAYOUT.grid[0] * LAYOUT.grid[1]:
            raise ValueError("base support must cover each native region")
        max_base = HEADER_BYTES + sum(RECORD_BYTES + ((b - a) * self.bits + 7) // 8
                                      for a, b in zip(starts, starts[1:len(starts_base) + 1]))
        if self.base_cap is not None and max_base > self.base_cap:
            raise ValueError("worst-case base exceeds the Profile's physical byte ceiling")
        fields = {"scan": scan, "scan_tokens": _immutable(tokens, "<i4"), "starts": starts,
                  "native_base_order": _immutable(base_order, "<i4"),
                  "scan_base_order": _immutable(np.argsort(base_order), "<i4"),
                  "intra_steps": steps_i, "inter_steps": steps_p,
                  "steps_i": _immutable(steps_i[rank, components], "<i4"),
                  "steps_p": _immutable(steps_p[rank[:base], components[:base]], "<i4"),
                  "base_support": base_support.tobytes(), "base_records": len(starts_base), "max_base_bytes": max_base}
        for key, value in fields.items():
            object.__setattr__(self, key, value)
        object.__setattr__(self, "binding", self._identity())

    def with_predictor(self, predictor: str, *, refresh: int, scene_threshold: float | None = None) -> "Profile":
        return replace(self, predictor=predictor, refresh=refresh, scene_threshold=scene_threshold)

    def mode_allowed(self, mode: Mode) -> bool:
        if mode == Mode.RAW8:
            return self.bits == 8
        if mode == Mode.RAW4:
            return self.bits == 4 or self.sparse
        return self.sparse

    def end(self, records: int) -> int:
        return self.starts[records]

    @property
    def records(self) -> int:
        return len(self.starts) - 1
