"""Instrument the pinned official DCVC-UF CUDA codec and NAL path."""

from hashlib import sha256
from pathlib import Path
from time import perf_counter
from typing import Any

import torch
from torch import Tensor

from hla_tok.hts.units import Unit, pad_indices
from hla_tok.uf.base import BaseFrame, FrameInfo
from third_party.DCVC.src.models.image_model import DMCI
from third_party.DCVC.src.models.video_model_ht import DMC as HT
from third_party.DCVC.src.models.video_model_ld import DMC as LD
from third_party.DCVC.src.utils.common import ModelStructure, get_state_dict
from third_party.DCVC.src.utils.transforms import rgb2ycbcr, ycbcr2rgb


def frame_input(rgb: Tensor, unit: Unit) -> Tensor:
    frames = rgb[list(pad_indices(unit))]
    yuv = rgb2ycbcr(frames).reshape(1, 3 * unit.coded, *frames.shape[-2:])
    return (yuv.half() - 0.5).contiguous(memory_format=torch.channels_last)


def restore_native_rgb(yuv: list[Tensor], unit: Unit, height: int, width: int) -> Tensor:
    decoded = torch.cat(yuv[:unit.valid])[..., :height, :width]
    return ycbcr2rgb(decoded + 0.5).float()


def tensor_digest(tensors: list[Tensor]) -> str:
    digest = sha256()
    for tensor in tensors:
        digest.update(str((tuple(tensor.shape), tensor.dtype)).encode())
        digest.update(tensor.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy())
    return digest.hexdigest()


def load_weights(net, path: Path, skip_threshold: float):
    net.eval().requires_grad_(False)
    net.load_state_dict(get_state_dict(path), strict=True)
    net.update(skip_threshold)
    return net.half().cuda().to(memory_format=torch.channels_last)


class NativeNetworks:
    """The same model instances used by the official encode-then-decode loop."""

    def __init__(self, weights: Path, variant: str, skip_threshold: float):
        self.intra = load_weights(DMCI(), weights / "cvpr2026_image.pth.tar", skip_threshold)
        video = LD() if variant == 'ld' else HT(ModelStructure(variant))
        self.video = load_weights(video, weights / f"cvpr2026_video_{variant}.pth.tar", skip_threshold)

    def encode(self, rgb: Tensor, unit: Unit, qp: int) -> BaseFrame:
        height, width = rgb.shape[-2:]
        right, bottom = DMCI.get_padding_size(height, width, 16)
        source = frame_input(rgb, unit)
        if unit.intra:
            encoded = self.intra.compress(source, qp, bottom, right)
            self.video.clear_dpb()
            self.video.add_ref_feature_from_frame(encoded["x_hat"])
        else:
            encoded = self.video.compress(source, qp, unit.reset, bottom, right)
        info = FrameInfo(unit.start, height, width, qp, unit.intra, encoded['ec_parallel'], unit.reset)
        return BaseFrame(info, encoded['bit_stream'])

    def decode(self, base: BaseFrame) -> list[Tensor]:
        info = base.info
        sps = {'height': info.height, 'width': info.width}
        if not info.intra:
            result = self.video.decompress(base.payload, sps, info.qp, info.parallel, info.reset)["x_hat"]
            return result if isinstance(result, list) else [result]
        x = self.intra.decompress(base.payload, sps, info.qp, info.parallel)["x_hat"]
        self.video.clear_dpb()
        self.video.add_ref_feature_from_frame(x, apply_feature_adaptor=False)
        return [x]

    def reference_hash(self, unit: Unit) -> str:
        return tensor_digest(self.video.proxy.audit_reference(unit.intra or unit.reset))


def timed(call, *args: Any):
    torch.cuda.synchronize()
    begin = perf_counter()
    result = call(*args)
    torch.cuda.synchronize()
    return result, (perf_counter() - begin) * 1000
