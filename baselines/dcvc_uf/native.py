"""Instrument the pinned official DCVC-UF CUDA codec and NAL path."""

from pathlib import Path

import torch
from torch import Tensor

from baselines.dcvc_uf.frames import BaseFrame, FrameInfo
from baselines.dcvc_uf.units import Unit, pad_indices
from common.integrity import tensor_digest
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




def load_weights(net, path: Path, skip_threshold: float):
    net.eval().requires_grad_(False)
    net.load_state_dict(get_state_dict(path), strict=True)
    net.update(skip_threshold)
    return net.half().cuda().to(memory_format=torch.channels_last)


def decode_intra(model, base: BaseFrame) -> Tensor:
    if model.proxy is None:
        from inference_extensions_cuda import DMCIProxy

        model.proxy = DMCIProxy()
        model.proxy.set_param(model.add_cdf_to_state_dict(model.state_dict()), model.gaussian_encoder.skip_thres)
    info = base.info
    sps = {"height": info.height, "width": info.width}
    return model.decompress(base.payload, sps, info.qp, info.parallel)["x_hat"]


class NativeNetworks:
    """The same model instances used by the official encode-then-decode loop."""

    def __init__(self, weights: Path, variant: str, skip_threshold: float):
        self.intra = load_weights(DMCI(), weights / "cvpr2026_image.pth.tar", skip_threshold)
        video = LD() if variant == 'ld' else HT(ModelStructure(variant))
        self.video = load_weights(video, weights / f"cvpr2026_video_{variant}.pth.tar", skip_threshold)

    def encode(self, rgb: Tensor, unit: Unit, qp: int) -> BaseFrame:
        return self._encode(frame_input(rgb, unit), unit, qp)

    def encode_centered(self, frames: Tensor, unit: Unit, qp: int) -> BaseFrame:
        if len(frames) != unit.valid:
            raise ValueError("frames must match the native coding unit")
        tail = frames[-1:].expand(unit.coded - unit.valid, -1, -1, -1)
        source = torch.cat((frames, tail)).reshape(1, 3 * unit.coded, *frames.shape[-2:])
        return self._encode(source.contiguous(memory_format=torch.channels_last), unit, qp)

    def _encode(self, source: Tensor, unit: Unit, qp: int) -> BaseFrame:
        height, width = source.shape[-2:]
        right, bottom = DMCI.get_padding_size(height, width, 16)
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
        x = decode_intra(self.intra, base)
        self.video.clear_dpb()
        self.video.add_ref_feature_from_frame(x, apply_feature_adaptor=False)
        return [x]

    def reference_hash(self, unit: Unit) -> str:
        return tensor_digest(self.video.proxy.audit_reference(unit.intra or unit.reset))
