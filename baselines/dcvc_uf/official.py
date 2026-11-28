"""Read and write the pinned DCVC-UF I/P and sequence-parameter NAL units."""

from io import BytesIO

from baselines.dcvc_uf.frames import BaseFrame, FrameInfo
from third_party.DCVC.src.utils.stream_helper import (
    NalType,
    read_header,
    read_ip_remaining,
    read_sps_remaining,
    write_ip,
    write_sps,
)


def _sps(info: FrameInfo) -> dict:
    return {"sps_id": 0, "height": info.height, "width": info.width}


class NativeNAL:
    def __init__(self) -> None:
        self.sps: dict | None = None

    def encode(self, frame: BaseFrame) -> bytes:
        info = frame.info
        output = BytesIO()
        sps = _sps(info)
        if self.sps is None:
            write_sps(output, sps)
            self.sps = sps
        elif self.sps != sps:
            raise ValueError("SPS resolution changed inside a fixed-resolution sequence")
        write_ip(output, info.intra, 0, info.qp, info.parallel, info.reset, frame.payload)
        return output.getvalue()

    def decode(self, raw: bytes, index: int) -> BaseFrame:
        source = BytesIO(raw)
        header = read_header(source)
        if header["nal_type"] == NalType.NAL_SPS:
            self.sps = read_sps_remaining(source, header["sps_id"])
            header = read_header(source)
        if header["nal_type"] not in (NalType.NAL_I, NalType.NAL_P):
            raise ValueError("expected one native I/P access unit")
        if self.sps is None or header["sps_id"] != self.sps["sps_id"]:
            raise ValueError("missing native sequence parameter set")
        qp, parallel, reset, payload = read_ip_remaining(source)
        if source.read(1):
            raise ValueError("unexpected trailing NAL data")
        info = FrameInfo(index, self.sps["height"], self.sps["width"], qp,
                         header["nal_type"] == NalType.NAL_I, parallel, bool(reset))
        return BaseFrame(info, payload)
