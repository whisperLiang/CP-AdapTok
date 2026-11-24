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
