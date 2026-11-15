"""Finite RD candidate scan online; whole-sequence byte-constrained MILP offline."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class RDPoint:
    tokens: int
    bytes: int
    distortion: float

    def __post_init__(self):
        if self.tokens <= 0 or self.bytes <= 0 or not math.isfinite(self.distortion) or self.distortion < 0:
            raise ValueError("invalid rate-distortion point")


class OnlineTokenAllocator:
    def __init__(self, k_ref=128):
        self.k_ref = k_ref

    def allocate(self, points, bandwidth_bps, deadline_s, queued_bytes=0, buffer_s=0,
                 mode="scorer", fixed_k=None):
        if bandwidth_bps <= 0 or min(deadline_s, queued_bytes, buffer_s) < 0:
            raise ValueError("invalid bandwidth/deadline/queue/buffer")
        if mode not in ("fixed", "rate-only", "scorer"):
            raise ValueError("unknown online allocation mode")
        available = max(0, (deadline_s + buffer_s) * bandwidth_bps / 8 - queued_bytes)
        feasible = [p for p in points if p.tokens >= self.k_ref and p.bytes <= available]
        if mode == "fixed":
            feasible = [p for p in feasible if p.tokens == fixed_k]
        if not feasible:
            return None  # base must wait for reliable delivery, never silently skip its reference
        if mode == "rate-only":
            return max(feasible, key=lambda p: p.tokens)
        return min(feasible, key=lambda p: (p.distortion, p.bytes))

    @staticmethod
    def enhancement_priority(points):
        """Marginal quality per byte; dependencies still require contiguous transmission."""
        ordered = sorted(points, key=lambda p: p.tokens)
        return [(b.tokens, (a.distortion - b.distortion) / max(1, b.bytes - a.bytes))
                for a, b in zip(ordered, ordered[1:])]


def offline_oracle(frames, total_bytes):
    """Noncausal oracle: reads all frames' measured distortions. Never called by the sender."""
    from scipy.optimize import Bounds, LinearConstraint, milp
    points = [point for frame in frames for point in frame]
    costs = np.asarray([p.bytes for p in points], dtype=float)
    matrix = np.zeros((len(frames), len(points)))
    offset = 0
    for i, frame in enumerate(frames):
        matrix[i, offset:offset + len(frame)] = 1
        offset += len(frame)
    result = milp(np.asarray([p.distortion for p in points]), integrality=np.ones(len(points)),
                  bounds=Bounds(0, 1), constraints=[LinearConstraint(matrix, 1, 1),
                                                   LinearConstraint(costs, 0, total_bytes)],
                  options={"time_limit": 30})
    if not result.success:
        raise ValueError("offline oracle is infeasible or solver did not converge")
    return [p for p, selected in zip(points, result.x) if selected > .5]
