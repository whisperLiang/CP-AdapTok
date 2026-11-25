"""Independent byte-shuffled BF16 blocks as a diagnostic rate reference."""

from collections.abc import Callable
from pathlib import Path
from struct import Struct
from time import perf_counter
from zlib import crc32

import numpy as np

from hla_tok.compression.packing import pack_bf16
from hla_tok.evaluation.io import write_csv, write_json
from hla_tok.streaming.continuous import MAX_K
from scripts.hla_tok.compact.cache import load_train_tokens
from scripts.hla_tok.compact.inputs import OUTPUT, compact_inputs
from scripts.hla_tok.continuous.inputs import Inputs, locked_output, run_arguments
from scripts.hla_tok.continuous.matrix import groups

ROW_BYTES = 64
BLOCK_HEADER = Struct("<HHII")
Compress = Callable[[bytes], bytes]
Decompress = Callable[[bytes, int], bytes]


def encode_block(rows: np.ndarray, start: int, compress: Compress) -> bytes:
    if rows.shape != (len(rows), ROW_BYTES) or not 0 < len(rows) <= MAX_K - start:
        raise ValueError("invalid lossless native Token range")
    body = compress(rows.T.tobytes())
    return BLOCK_HEADER.pack(start, len(rows), len(body), crc32(body)) + body


def decode_block(raw: bytes, decompress: Decompress) -> tuple[int, bytes]:
    if len(raw) < BLOCK_HEADER.size:
        raise ValueError("truncated lossless block")
    start, count, length, digest = BLOCK_HEADER.unpack_from(raw)
    body = raw[BLOCK_HEADER.size:]
    if not 0 < count <= MAX_K - start or length != len(body) or crc32(body) != digest:
        raise ValueError("lossless block header or CRC32 mismatch")
    decoded = np.frombuffer(decompress(body, count * ROW_BYTES), dtype=np.uint8)
    if decoded.size != count * ROW_BYTES:
        raise ValueError("decoded block length differs")
    return start, decoded.reshape(ROW_BYTES, count).T.tobytes()


def _codecs() -> dict[str, tuple[Compress, Decompress]]:
    import lz4.block
    import zstandard

    zstd = zstandard.ZstdCompressor(level=3, write_content_size=False)
    restore = zstandard.ZstdDecompressor()
    return {
        "shuffle_lz4": (lambda data: lz4.block.compress(data, store_size=False),
                        lambda data, size: lz4.block.decompress(data, uncompressed_size=size)),
        "shuffle_zstd": (zstd.compress, lambda data, size: restore.decompress(data, max_output_size=size)),
    }


def _frame(raw: bytes, block: int, encode: Compress, decode: Decompress) -> dict:
    rows = np.frombuffer(raw, dtype=np.uint8).reshape(-1, ROW_BYTES)
    start = perf_counter()
    packets = [encode_block(rows[i:i + block], i, encode) for i in range(0, len(rows), block)]
    enc_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    parts = [decode_block(packet, decode) for packet in packets]
    recovered = b"".join(part for _, part in parts)
    dec_ms = (perf_counter() - start) * 1000
    if recovered != raw or [at for at, _ in parts] != list(range(0, len(rows), block)):
        raise RuntimeError("lossless block contents or original offsets differ")
    wire = sum(map(len, packets))
    heads = len(packets) * BLOCK_HEADER.size
    return {"raw_bytes": len(raw), "payload_bytes": wire - heads, "header_bytes": heads,
            "total_bytes": wire, "blocks": len(packets), "encode_ms": enc_ms, "decode_ms": dec_ms}


def run(inputs: Inputs, config: dict, output: Path) -> None:
    cache = load_train_tokens(inputs, config, output)
    blocks = config["calibration"]["lossless_block_tokens"]
    methods = _codecs()
    records = []
    for index, vector in enumerate(cache.native().cpu()):
        start = perf_counter()
        raw = pack_bf16(vector)
        enc_ms = (perf_counter() - start) * 1000
        start = perf_counter()
        verified = np.frombuffer(raw, dtype="<u2")
        _ = verified.reshape(-1, vector.shape[-1]).copy()
        dec_ms = (perf_counter() - start) * 1000
        records.append({"method": "raw_bf16", "block_tokens": 1, "frame": index,
                        "raw_bytes": len(raw), "payload_bytes": len(raw), "header_bytes": 0,
                        "total_bytes": len(raw), "blocks": 0, "encode_ms": enc_ms, "decode_ms": dec_ms})
        for name, (encode, decode) in methods.items():
            for size in blocks:
                records.append({"method": name, "block_tokens": size, "frame": index} | _frame(raw, size, encode, decode))
    write_csv(output / "lossless_per_frame.csv", records)
    summary = []
    for keys, rows in groups(records, ("method", "block_tokens")):
        values = {key: sum(row[key] for row in rows) / len(rows)
                  for key in ("raw_bytes", "payload_bytes", "header_bytes", "total_bytes", "encode_ms", "decode_ms")}
        summary.append(keys | values | {"frames": len(rows), "raw_over_wire_ratio": values["raw_bytes"] / values["total_bytes"],
                                        "strict_one_token_prefix": keys["method"] == "raw_bf16"})
    write_csv(output / "lossless_diagnostics.csv", summary)
    write_json(output / "lossless_audit.json", {"status": "PASS", "split": "train", "independent_blocks": True,
               "incremental_transmission": "only send new complete blocks; final partial block is legal",
               "raw_rate_scope": "unframed BF16 reference", "block_header_bytes": BLOCK_HEADER.size,
               "test_pixel_reads": inputs.reader.counts["test"]})


def main() -> None:
    parser = run_arguments(__doc__)
    parser.set_defaults(output=OUTPUT)
    args = parser.parse_args()
    with locked_output(args.output):
        inputs, config = compact_inputs(args.v3, args.output)
        run(inputs, config, args.output)


if __name__ == "__main__":
    main()
