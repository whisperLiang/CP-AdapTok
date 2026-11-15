import torch
from torch import nn
from torch.nn import functional as F

from third_party.DCVC.src.models.image_model import DMCI
from third_party.DCVC.src.utils.common import get_state_dict
from third_party.DCVC.src.utils.transforms import rgb2ycbcr, ycbcr2rgb


def prepare_rgb(rgb):
    if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point():
        raise ValueError("expected floating RGB [B,3,H,W] in [0,1]")
    if not torch.isfinite(rgb).all() or rgb.min() < 0 or rgb.max() > 1:
        raise ValueError("RGB must be finite and in [0,1]")
    h, w = rgb.shape[-2:]
    centered = rgb2ycbcr(rgb) - .5
    return F.pad(centered, (0, (-w) % 64, 0, (-h) % 64), mode="replicate")


def restore_rgb(centered_yuv, height, width):
    return ycbcr2rgb((centered_yuv[..., :height, :width] + .5).clamp(0, 1)).clamp(0, 1)


class IntraBackboneAdapter(nn.Module):
    latent_channels = 256

    def __init__(self, checkpoint=None):
        super().__init__()
        self.model = DMCI()
        if checkpoint:
            self.model.load_state_dict(get_state_dict(checkpoint), strict=True)

    def quant(self, name, qp):
        parameter = getattr(self.model, name)
        index = torch.tensor([qp], device=parameter.device, dtype=torch.long)
        return self.model.index_select_dim0(parameter, index)

    def encode_latent(self, x_t, frozen_context):
        return self.model.enc(x_t, self.quant("q_scale_enc", frozen_context.qp))

    def decode_latent(self, y_hat, frozen_context):
        yuv = self.model.dec(y_hat, self.quant("q_scale_dec", frozen_context.qp)).clamp(-.5, .5)
        return yuv, F.pixel_unshuffle(yuv, 8)

    @staticmethod
    def commit_base_state(feature, frozen_context, base_digest):
        return frozen_context.commit(feature, base_digest)
