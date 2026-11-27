"""Unmodified official UF-LD/Intra FP16 inference and native NAL framing."""

import json
import subprocess
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import torch
from torch import Tensor

from baselines.dcvc_uf.frames import BaseFrame, FrameInfo, ReferenceOrder
from common.integrity import sha256_file
from common.paths import ROOT
from common.video import restore_rgb
from third_party.DCVC.src.models.image_model import DMCI
from third_party.DCVC.src.models.video_model_ld import DMC, g_frame_delay
from third_party.DCVC.src.utils.common import get_state_dict
from third_party.DCVC.src.utils.stream_helper import (
    NalType,
    read_header,
    read_ip_remaining,
    read_sps_remaining,
    write_ip,
    write_sps,
)
from third_party.DCVC.src.utils.transforms import rgb2ycbcr

UPSTREAM = ROOT / "third_party/DCVC"
LOCK = ROOT / "configs/baselines/dcvc_uf.json"




@dataclass(frozen=True, slots=True)
class OfficialAssets:
    intra: Path
    ld: Path

    @classmethod
    def verify(cls, directory: Path) -> "OfficialAssets":
        lock = json.loads(LOCK.read_text())
        git = ["git", "-C", str(UPSTREAM)]
        commit = subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()
        changed = subprocess.check_output([*git, "status", "--porcelain", "--untracked-files=no"], text=True)
        if commit != lock["commit"] or changed or g_frame_delay != 1 or lock["frame_delay"] != 1:
            raise ValueError("official UF-LD source is not the pinned, clean N=1 tree")
        paths = {key: _verified_file(directory, lock["weights"][key]) for key in ("intra", "ld")}
        return cls(**paths)


def _verified_file(directory: Path, expected: dict) -> Path:
    path = directory / expected["filename"]
    if path.stat().st_size != expected["bytes"] or sha256_file(path) != expected["sha256"]:
        raise ValueError(f"official weight hash or length differs: {path}")
    return path


def load_native_network(cls, path, proxy_cls, device):
    # D1 freezes the historical LD alignment (zero skipping, FP16, channels-last).
    net = cls().eval().to(device).half().to(memory_format=torch.channels_last)
    net.load_state_dict(get_state_dict(path), strict=True)
    net.update(skip_thres=0)
    net.proxy = proxy_cls()
    net.proxy.set_param(net.add_cdf_to_state_dict(net.state_dict()), net.gaussian_encoder.skip_thres)
    return net


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


class NativeEndpoint:
    """One writer owns each mutable CUDA graph, rANS worker, and DPB."""

    def __init__(self, assets: OfficialAssets, video: bool = True):
        from inference_extensions_cuda import DMCIProxy, DMCLDProxy

        self.stream = torch.cuda.Stream()
        with torch.cuda.stream(self.stream):
            self.intra = load_native_network(DMCI, assets.intra, DMCIProxy, "cuda")
            self.video = load_native_network(DMC, assets.ld, DMCLDProxy, "cuda") if video else None
        self.stream.synchronize()
        self.reset()

    def reset(self) -> None:
        self.order = ReferenceOrder()
        self.last_net: DMCI | DMC | None = None

    def _video(self) -> DMC:
        if self.video is None:
            raise ValueError("intra-only endpoint cannot process a P-frame")
        return self.video


    def _seed_video(self, decoded: Tensor, for_encode: bool) -> None:
        if self.video is not None:
            self.video.clear_dpb()
            self.video.add_ref_feature_from_frame(decoded, apply_feature_adaptor=for_encode)

    def decoded_latent(self) -> Tensor:
        if self.last_net is None:
            raise ValueError("decode a base frame before inspecting its latent")
        with torch.cuda.stream(self.stream):
            latent = self.last_net.proxy.get_decoded_latent()
        self.stream.synchronize()
        return latent


class NativeEncoder(NativeEndpoint):
    @torch.no_grad()
    def encode(self, rgb: Tensor, index: int, qp: int, intra: bool, reset: bool = False) -> BaseFrame:
        if rgb.ndim != 4 or rgb.shape[:2] != (1, 3) or not rgb.is_cuda:
            raise ValueError("expected one batch-1 CUDA RGB frame")
        height, width = rgb.shape[-2:]
        info = FrameInfo(index, height, width, qp, intra, 1, reset)
        self.order.check(info)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            encoded = self._compress(rgb, info)
        self.stream.synchronize()
        frame = BaseFrame(FrameInfo(index, *info.dimensions, qp, intra, encoded["ec_parallel"], reset),
                          encoded["bit_stream"])
        self.order.commit(frame.info)
        return frame

    def _compress(self, rgb: Tensor, info: FrameInfo) -> dict:
        x = (rgb2ycbcr(rgb) - 0.5).half().contiguous(memory_format=torch.channels_last)
        if not info.intra:
            return self._video().compress(x, info.qp, info.reset, *info.native_padding)
        encoded = self.intra.compress(x, info.qp, *info.native_padding)
        self._seed_video(encoded["x_hat"], for_encode=True)
        return encoded


class NativeDecoder(NativeEndpoint):
    @torch.no_grad()
    def decode(self, frame: BaseFrame) -> Tensor:
        self.order.check(frame.info)
        info = frame.info
        with torch.cuda.stream(self.stream):
            decoded = self._decompress(frame)
            rgb = restore_rgb(decoded.float(), *info.dimensions)
        self.stream.synchronize()
        self.order.commit(info)
        return rgb

    def _decompress(self, frame: BaseFrame) -> Tensor:
        info, payload = frame.info, frame.payload
        if not info.intra:
            video = self._video()
            self.last_net = video
            return video.decompress(payload, _sps(info), info.qp, info.parallel, info.reset)["x_hat"]
        decoded = self.intra.decompress(payload, _sps(info), info.qp, info.parallel)["x_hat"]
        self._seed_video(decoded, for_encode=False)
        self.last_net = self.intra
        return decoded
