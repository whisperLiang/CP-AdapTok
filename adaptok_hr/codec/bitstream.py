"""Version 1: checked frame headers and fixed-length big-endian 13-bit IDs."""

import struct
import zlib
from dataclasses import dataclass, replace

MAGIC = b"ATHR"
VERSION = 1
HEADER = struct.Struct(">4sBBIHHHHI16s32s32s32sII")


def pack_codes(ids):
    output = bytearray()
    pending, bits = 0, 0
    for code in ids:
        if type(code) is not int or not 0 <= code < 8192:
            raise ValueError("13-bit code ID must be an integer in [0,8191]")
        pending, bits = (pending << 13) | code, bits + 13
        while bits >= 8:
            bits -= 8
            output.append((pending >> bits) & 255)
        pending &= (1 << bits) - 1
    if bits:
        output.append((pending << (8 - bits)) & 255)
    return bytes(output)


def unpack_codes(payload, count):
    if type(count) is not int or count < 1 or len(payload) != (count * 13 + 7) // 8:
        raise ValueError("invalid payload length or count")
    output, pending, bits = [], 0, 0
    for byte in payload:
        pending, bits = (pending << 8) | byte, bits + 8
        if bits >= 13:
            bits -= 13
            output.append((pending >> bits) & 8191)
            pending &= (1 << bits) - 1
    if len(output) != count or pending:
        raise ValueError("noncanonical payload or nonzero padding bits")
    return output


@dataclass(frozen=True)
class Packet:
    frame_index: int
    intra: bool
    gop: int
    height: int
    width: int
    ids: tuple
    stream_id: bytes
    codebook: bytes
    model: bytes
    previous: bytes


def serialize(packet):
    payload = pack_codes(packet.ids)
    fields = (MAGIC, VERSION, int(packet.intra), packet.frame_index, packet.gop, packet.height,
              packet.width, len(packet.ids), len(payload), packet.stream_id, packet.codebook,
              packet.model, packet.previous, zlib.crc32(payload), 0)
    raw = HEADER.pack(*fields)
    raw = raw[:-4] + struct.pack(">I", zlib.crc32(raw[:-4]))
    return raw + payload


def checked_header(data):
    if not isinstance(data, bytes) or len(data) < HEADER.size:
        raise ValueError("incomplete frame header")
    fields = HEADER.unpack(data[:HEADER.size])
    magic, version, flags, index, gop, height, width, count, length = fields[:9]
    stream, codebook, model, previous, payload_crc, header_crc = fields[9:]
    if magic != MAGIC or version != VERSION or flags not in (0, 1):
        raise ValueError("unsupported bitstream identity/version/flags")
    if zlib.crc32(data[:HEADER.size - 4]) != header_crc:
        raise ValueError("corrupt frame header CRC")
    return fields


def validate_frame_fields(flags, index, gop, height, width, count):
    if gop < 1 or min(height, width, count) < 1 or count > 2685:
        raise ValueError("invalid frame dimensions, GOP or count")
    if flags != int(index % gop == 0):
        raise ValueError("frame type disagrees with GOP")


def parse(data):
    fields = checked_header(data)
    _, _, flags, index, gop, height, width, count, length = fields[:9]
    stream, codebook, model, previous, payload_crc, _ = fields[9:]
    validate_frame_fields(flags, index, gop, height, width, count)
    if length != (count * 13 + 7) // 8 or len(data) != HEADER.size + length:
        raise ValueError("incomplete or trailing frame payload")
    payload = data[HEADER.size:]
    if zlib.crc32(payload) != payload_crc:
        raise ValueError("corrupt payload CRC")
    ids = tuple(unpack_codes(payload, count))
    return Packet(index, bool(flags), gop, height, width, ids, stream, codebook, model, previous)


def truncate_packet(data, budget):
    packet = parse(data)
    if type(budget) is not int or not 1 <= budget <= len(packet.ids):
        raise ValueError("truncation budget must fit transmitted prefix")
    return serialize(replace(packet, ids=packet.ids[:budget]))
