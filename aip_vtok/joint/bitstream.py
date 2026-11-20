"""Versioned byte-prefix framing, with one independent rANS state per packet."""

import operator
import struct
import zlib
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from time import perf_counter

import torch

from aip_vtok.c1_followup.quantization import CARDINALITY, CODE_BITS

from .entropy import STATE, Tables, rans_decode, rans_encode
from .scan import AREA, BUDGETS, SLOT_COUNT, scan_digest
from .transform import frame_type

GOP = 32
PIXELS = 1920 * 1080
VERSION = 1
MAGIC = b"AIPJ"
FRAME_HEADER = struct.Struct("<4sBIB32s32s")
PACKET_FIELDS = struct.Struct("<IHH")
PACKET_HEADER = struct.Struct("<IHHI")
MAX_PACKET_TOKENS = 128


def packet_sizes() -> dict[int, int]:
    start, result = 0, {}
    for budget in BUDGETS:
        while start < budget:
            count = min(MAX_PACKET_TOKENS, budget - start)
            result[start] = count
            start += count
    return result


@dataclass(frozen=True)
class FrameBytes:
    data: bytes
    ends: dict[int, tuple[int, int, float]]

    def prefix(self, budget: int) -> bytes:
        return self.data[:self.ends[budget][0]]

    def ledger(self, budget: int) -> dict:
        size, packets, _ = self.ends[budget]
        framing = FRAME_HEADER.size + packets * PACKET_HEADER.size
        return {"transmitted_tokens": budget, "raw_token_bits": CODE_BITS * budget,
                "payload_bytes": size - framing, "rans_state_bytes": packets * STATE.size,
                "header_bytes": framing, "frame_header_bytes": FRAME_HEADER.size, "packet_count": packets,
                "total_bytes": size, "actual_bpp": 8 * size / PIXELS,
                "raw_bpp": CODE_BITS * budget / PIXELS, "entropy_bpp": 8 * (size - framing) / PIXELS,
                "overhead_bpp": 8 * framing / PIXELS}


class WireFormat:
    def __init__(self, table: torch.Tensor, tables: Tables, model_sha256: str):
        if table.shape != (SLOT_COUNT,) or not torch.equal(table.cpu().sort().values, torch.arange(SLOT_COUNT)):
            raise ValueError("scan table must be a complete slot permutation")
        self.groups = tuple((table.cpu() // AREA).tolist())
        self.packet_sizes = packet_sizes()
        self.tables = tables
        self.scan_hash = bytes.fromhex(scan_digest(table))
        self.model_hash = bytes.fromhex(model_sha256)
        if len(self.model_hash) != 32:
            raise ValueError("invalid model SHA256")

    def parse_header(self, data: bytes) -> int:
        if len(data) != FRAME_HEADER.size:
            raise ValueError("truncated frame header")
        magic, version, frame, intra, model, scan = FRAME_HEADER.unpack(data)
        if magic != MAGIC or version != VERSION:
            raise ValueError("invalid Joint bitstream version")
        if intra != int(frame % GOP == 0):
            raise ValueError("frame type differs from GOP phase")
        if model != self.model_hash or scan != self.scan_hash:
            raise ValueError("frame header model or scan identity mismatch")
        return frame

    def encode_frame(self, frame: int, ids: Sequence[int]) -> FrameBytes:
        start_time = perf_counter()
        if len(ids) != SLOT_COUNT:
            raise ValueError("encoding requires a complete ordered sequence")
        ids = tuple(map(operator.index, ids))
        if min(ids) < 0 or max(ids) >= CARDINALITY:
            raise ValueError("FSQ ID outside [0,64000)")
        intra = frame % GOP == 0
        tables = self.tables[frame_type(intra)]
        data = bytearray(FRAME_HEADER.pack(MAGIC, VERSION, frame, intra, self.model_hash, self.scan_hash))
        ends = {}
        for packet, (start, count) in enumerate(self.packet_sizes.items(), 1):
            end = start + count
            payload = rans_encode(ids[start:end], self.groups[start:end], tables)
            fields = PACKET_FIELDS.pack(start, count, len(payload))
            data.extend(fields)
            data.extend(struct.pack("<I", zlib.crc32(fields + payload)))
            data.extend(payload)
            ends[end] = (len(data), packet, (perf_counter() - start_time) * 1000)
        return FrameBytes(bytes(data), ends)

    def packets(self, data: bytes) -> Iterator[bytes]:
        start = FRAME_HEADER.size
        self.parse_header(data[:start])
        while start < len(data):
            if start + PACKET_HEADER.size > len(data):
                raise ValueError("truncated packet header")
            _, _, size, _ = PACKET_HEADER.unpack_from(data, start)
            end = start + PACKET_HEADER.size + size
            if end > len(data):
                raise ValueError("truncated packet payload")
            yield data[start:end]
            start = end

    def decode_packet(self, frame: int, packet: bytes) -> tuple[int, tuple[int, ...]]:
        if len(packet) < PACKET_HEADER.size:
            raise ValueError("truncated packet header")
        start, count, size, checksum = PACKET_HEADER.unpack_from(packet)
        payload = packet[PACKET_HEADER.size:]
        if self.packet_sizes.get(start) != count or size != len(payload):
            raise ValueError("invalid packet boundary or payload length")
        if zlib.crc32(packet[:PACKET_FIELDS.size] + payload) != checksum:
            raise ValueError("packet CRC mismatch")
        tables = self.tables[frame_type(frame % GOP == 0)]
        return start, rans_decode(payload, self.groups[start:start + count], tables)
