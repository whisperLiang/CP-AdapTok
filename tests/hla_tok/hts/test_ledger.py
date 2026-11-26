import pytest

from hla_tok.uf.base import BaseFrame, FrameInfo
from hla_tok.uf.official import NativeNAL
from scripts.hla_tok.hts.verify_ledger import verify_stream


def test_native_ledger_accounts_for_nal_bytes_and_rejects_a_trailer():
    base = BaseFrame(FrameInfo(15, 1080, 1920, 5, True, 1, False), bytes(range(32)))
    packet = NativeNAL().encode(base)
    row = {"total_bytes": len(packet), "native_payload_bytes": len(base.payload),
           "header_bytes": len(packet) - len(base.payload), "frame_ids": [15], "valid_frames": 1,
           "offset": 0, "qp_effective": 5, "memory_reset": False, "unit_type": "I"}
    verify_stream(packet, [row])
    with pytest.raises(ValueError, match="trailing bytes"):
        verify_stream(packet + b"extra", [row])
    with pytest.raises(ValueError, match="payload do not sum"):
        verify_stream(packet, [row | {"header_bytes": 0}])
