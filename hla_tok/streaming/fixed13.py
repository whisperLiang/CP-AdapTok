"""Little-endian 13-bit index payloads for the Phase 0 baseline rate ledger."""

from collections.abc import Iterable
from operator import index

TOKEN_BITS = 13
TOKEN_MASK = (1 << TOKEN_BITS) - 1


def pack_tokens_13bit(ids: Iterable[int]) -> bytes:
    output = bytearray()
    word, bits = 0, 0
    for symbol in ids:
        symbol = index(symbol)
        if not 0 <= symbol <= TOKEN_MASK:
            raise ValueError("13-bit token outside the codebook")
        word |= symbol << bits
        bits += TOKEN_BITS
        while bits >= 8:
            output.append(word & 0xff)
            word >>= 8
            bits -= 8
    if bits:
        output.append(word)
    return bytes(output)


def unpack_tokens_13bit(payload: bytes, count: int) -> list[int]:
    if type(count) is not int or count < 0 or len(payload) != (count * TOKEN_BITS + 7) // 8:
        raise ValueError("13-bit payload length does not match token count")
    output = []
    word, bits = 0, 0
    for byte in payload:
        word |= byte << bits
        bits += 8
        if bits >= TOKEN_BITS:
            output.append(word & TOKEN_MASK)
            word >>= TOKEN_BITS
            bits -= TOKEN_BITS
    if word:
        raise ValueError("13-bit payload padding must be zero")
    return output


def payload_ledger(byte_count: int, tokens: int, frames: int, height: int, width: int) -> dict:
    """Account for actual Fixed13 buffers, including every partial final byte."""
    pixels = frames * height * width
    padding_bits = byte_count * 8 - tokens * TOKEN_BITS
    if pixels <= 0 or padding_bits < 0:
        raise ValueError("invalid Fixed13 byte or geometry ledger")
    return {"total_bytes": byte_count, "padding_bits": padding_bits, "token_per_frame": tokens / frames,
            "actual_bpp": byte_count * 8 / pixels, "rate_basis": "actual_fixed13_payload"}
