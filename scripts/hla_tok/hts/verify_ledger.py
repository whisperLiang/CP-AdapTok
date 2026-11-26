"""Reopen actual native streams and verify the byte and valid-frame ledger."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from hla_tok.evaluation.io import write_json
from hla_tok.uf.official import NativeNAL


def verify_stream(raw: bytes, rows: list[dict]) -> None:
    parser = NativeNAL()
    cursor = 0
    previous_end = rows[0]["frame_ids"][0]
    for row in rows:
        stop = cursor + row["total_bytes"]
        ids = row["frame_ids"]
        if ids != list(range(previous_end, previous_end + row["valid_frames"])):
            raise ValueError("source frame coverage in the ledger has a gap")
        if row["offset"] != cursor or stop > len(raw):
            raise ValueError("unit exceeds or disagrees with its byte range")
        packet = parser.decode(raw[cursor:stop], ids[0])
        if len(packet.payload) != row["native_payload_bytes"] or stop - cursor - len(packet.payload) != row["header_bytes"]:
            raise ValueError("native header and payload do not sum to physical bytes")
        if (packet.info.qp, packet.info.intra, packet.info.reset) != (
                row["qp_effective"], row["unit_type"] == "I", row["memory_reset"]):
            raise ValueError("parsed native fields disagree with the ledger")
        cursor, previous_end = stop, ids[-1] + 1
    if cursor != len(raw):
        raise ValueError("unaccounted trailing bytes in native stream")


def verify(directory: Path) -> dict:
    counts = {"streams": 0, "units": 0, "bytes": 0, "valid_frames": 0, "padded_frames": 0}
    for path in sorted(directory.glob("*/*/point.json")):
        point = json.loads(path.read_text())
        streams: dict[str, list[dict]] = defaultdict(list)
        for row in point["units"]:
            streams[row["stream"]].append(row)
        total = 0
        for filename, rows in streams.items():
            source = (directory / filename).resolve()
            if not source.is_relative_to(path.parent.resolve()):
                raise ValueError(f"native audit stream points outside its sweep point: {filename}")
            raw = source.read_bytes()
            verify_stream(raw, rows)
            total += len(raw)
        if total != point["summary"]["total_bytes"]:
            raise ValueError("aggregate did not charge all native stream bytes")
        counts["streams"] += len(streams)
        counts["units"] += len(point["units"])
        counts["bytes"] += total
        counts["valid_frames"] += point["summary"]["frames"]
        counts["padded_frames"] += point["summary"]["padded_frames"]
    if not counts["streams"]:
        raise ValueError("missing physical native streams")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    result = verify(args.directory)
    write_json(args.directory / "byte_accounting.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
