"""Only fully decoded base coordinates and their mask are retained for prediction."""

from dataclasses import dataclass
from hashlib import sha256
from struct import pack

import numpy as np

from hla_tok.streaming.temporal_wire import FrameHeader
from hla_tok.temporal.profile import Profile
from hla_tok.temporal.quantization import LATTICE, reconstruct


@dataclass(frozen=True)
class BaseReferenceState:
    frame_id: int
    last_intra: int
    profile_binding: bytes
    digest: bytes
    _coordinates: bytes
    _support: bytes
    _previous: bytes

    @property
    def coordinates(self) -> np.ndarray:
        return np.frombuffer(self._coordinates, dtype="<i2")

    @property
    def support(self) -> np.ndarray:
        return np.frombuffer(self._support, dtype=bool)


@dataclass(frozen=True)
class DisplayReconstructionState:
    coordinates: np.ndarray
    support: np.ndarray
    records: int


def use_intra(frame_id: int, reference: BaseReferenceState | None, profile: Profile, target: np.ndarray,
              force: bool) -> bool:
    if force or reference is None or profile.predictor == "absolute":
        return True
    if frame_id - reference.last_intra >= profile.refresh:
        return True
    if profile.scene_threshold is None:
        return False
    current = target[profile.scan[:profile.base_components][profile.native_base_order]].astype(np.int32)
    previous = reference.coordinates
    distance = float(np.abs(current - previous).mean())
    energy = max(float(np.abs(current).mean()), 1.0)
    return distance > profile.scene_threshold * energy


def predict(profile: Profile, reference: BaseReferenceState | None, intra: bool) -> np.ndarray | int:
    if intra:
        return 0
    if reference is None:
        raise ValueError("P frame requires a committed base")
    previous = reference.coordinates[profile.scan_base_order]
    if profile.predictor != "linear" or not reference._previous:
        return previous
    older = np.frombuffer(reference._previous, dtype="<i2")[profile.scan_base_order]
    bound = profile.linear_bound_steps * profile.steps_p
    delta = (previous.astype(np.int32) - older).clip(-bound, bound)
    return (previous.astype(np.int32) + delta).clip(LATTICE.min, LATTICE.max).astype("<i2")


def commit(profile: Profile, header: FrameHeader, reference: BaseReferenceState | None,
           prediction: np.ndarray | int, codes: np.ndarray) -> BaseReferenceState:
    step = profile.steps_i[:profile.base_components] if header.intra else profile.steps_p
    coordinates = reconstruct(prediction, codes, step)[profile.native_base_order].tobytes()
    last_intra = header.frame_id if header.intra or reference is None else reference.last_intra
    previous = (reference._coordinates if not header.intra and reference is not None and profile.predictor == "linear"
                else b"")
    digest = sha256(header.binding + pack("<II", header.frame_id, last_intra))
    for piece in (profile.base_support, coordinates, previous):
        digest.update(piece)
    return BaseReferenceState(header.frame_id, last_intra, header.binding, digest.digest(), coordinates,
                              profile.base_support, previous)
