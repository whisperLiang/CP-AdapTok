"""Version 1 wire format. Fixed big-endian headers, CRC32, bounded allocations."""

import struct
import zlib
from dataclasses import dataclass

FRAME = struct.Struct(">4sBBBBQqHHHHHHHH16s16s16s16s")
PACKET = struct.Struct(">4sQIHHI")
CRC = struct.Struct(">I")
FRAME_HEADER_BYTES = FRAME.size + CRC.size
PACKET_HEADER_BYTES = PACKET.size + CRC.size
MAX_RECORD_BYTES = 1 << 20


class StreamError(ValueError):
    pass


def _checked_body(wire):
    if len(wire) < 4 or CRC.unpack(wire[-4:])[0] != zlib.crc32(wire[:-4]):
        raise StreamError("CRC mismatch or truncated record")
    return wire[:-4]


def _crc_record(body):
    return body + CRC.pack(zlib.crc32(body))


@dataclass(frozen=True)
class FrameHeader:
    frame_id: int
    frame_type: str
    reference_id: int
    height: int
    width: int
    n_max: int
    k_ref: int
    packet_tokens: int
    codebook_size: int
    model_id: str
    codebook_id: str
    entropy_id: str
    reference_fingerprint: str = "00" * 16
    qp: int = 32

    def __post_init__(self):
        self._validate_frame_identity()
        self._validate_token_shape()
        self._validate_reference()
        self._validate_identifiers()

    def _validate_frame_identity(self):
        if self.frame_type not in ("I", "P") or not 0 <= self.qp < 64 or self.frame_id < 0:
            raise StreamError("invalid frame type/id/quantization")
        if not (1 <= self.height <= 8192 and 1 <= self.width <= 8192):
            raise StreamError("unsupported dimensions")

    def _validate_token_shape(self):
        if not (1 <= self.k_ref <= self.n_max <= 8192 and self.packet_tokens in (8, 16, 32, 64, 128, 256)):
            raise StreamError("invalid token shape")
        if self.n_max % self.packet_tokens or self.k_ref % self.packet_tokens:
            raise StreamError("N_max and K_ref must end on packet boundaries")
        if not 2 <= self.codebook_size <= 32768:
            raise StreamError("invalid alphabet")

    def _validate_reference(self):
        if (self.frame_type == "I" and self.reference_id != -1) or (
            self.frame_type == "P" and self.reference_id != self.frame_id - 1
        ):
            raise StreamError("invalid reference id")

    def _validate_identifiers(self):
        for value in (self.model_id, self.codebook_id, self.entropy_id, self.reference_fingerprint):
            if len(value) != 32:
                raise StreamError("identifiers must contain 128 bits")
            try:
                bytes.fromhex(value)
            except ValueError as error:
                raise StreamError("invalid identifier") from error

    @property
    def padded_height(self):
        return (self.height + 63) // 64 * 64

    @property
    def padded_width(self):
        return (self.width + 63) // 64 * 64

    @property
    def latent_shape(self):
        return self.padded_height // 16, self.padded_width // 16

    def to_bytes(self):
        body = FRAME.pack(
            b"CPAT",
            1,
            int(self.frame_type == "P"),
            self.qp,
            1,
            self.frame_id,
            self.reference_id,
            self.height,
            self.width,
            self.padded_height,
            self.padded_width,
            self.n_max,
            self.k_ref,
            self.packet_tokens,
            self.codebook_size,
            bytes.fromhex(self.model_id),
            bytes.fromhex(self.codebook_id),
            bytes.fromhex(self.entropy_id),
            bytes.fromhex(self.reference_fingerprint),
        )
        return _crc_record(body)

    @classmethod
    def from_bytes(cls, wire):
        if len(wire) != FRAME_HEADER_BYTES:
            raise StreamError("invalid frame header length")
        f = FRAME.unpack(_checked_body(wire))
        if f[0:2] != (b"CPAT", 1) or f[2] not in (0, 1) or f[4] != 1:
            raise StreamError("unsupported stream/color/model format version")
        header = cls(
            f[5],
            "P" if f[2] else "I",
            f[6],
            f[7],
            f[8],
            f[11],
            f[12],
            f[13],
            f[14],
            f[15].hex(),
            f[16].hex(),
            f[17].hex(),
            f[18].hex(),
            f[3],
        )
        if (f[9], f[10]) != (header.padded_height, header.padded_width):
            raise StreamError("noncanonical padding")
        return header


@dataclass(frozen=True)
class Packet:
    frame_id: int
    first_token_index: int
    token_count: int
    block_id: int
    payload: bytes

    def to_bytes(self):
        return _crc_record(
            PACKET.pack(
                b"CPTK",
                self.frame_id,
                self.first_token_index,
                self.token_count,
                self.block_id,
                len(self.payload),
            )
            + self.payload
        )

    @classmethod
    def from_bytes(cls, wire):
        if not PACKET_HEADER_BYTES <= len(wire) <= MAX_RECORD_BYTES:
            raise StreamError("invalid packet length")
        body = _checked_body(wire)
        magic, frame, first, count, block, size = PACKET.unpack_from(body)
        if magic != b"CPTK" or size != len(body) - PACKET.size or count == 0:
            raise StreamError("invalid packet metadata")
        return cls(frame, first, count, block, body[PACKET.size :])
