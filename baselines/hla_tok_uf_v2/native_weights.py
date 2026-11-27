"""Bounded P-tail updates folded into the official native proxy parameters."""

from hashlib import sha256

import torch
from torch import Tensor, nn

from baselines.dcvc_uf.official import NativeEndpoint

LATENT_CHANNELS = 128
PIXEL_CHANNELS = 192
GAIN_LIMIT = 0.035
PROBABILITY_LIMIT = 0.12


class NativeAdjustments(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_logits = nn.Parameter(torch.zeros(LATENT_CHANNELS))
        self.reconstruction_logits = nn.Parameter(torch.zeros(PIXEL_CHANNELS))
        self.hyper_logits = nn.Parameter(torch.zeros(LATENT_CHANNELS))

    def encoder_gain(self) -> Tensor:
        return 1 + GAIN_LIMIT * self.encoder_logits.tanh()

    def reconstruction_gain(self) -> Tensor:
        return 1 + GAIN_LIMIT * self.reconstruction_logits.tanh()

    def hyper_bias(self) -> Tensor:
        return PROBABILITY_LIMIT * self.hyper_logits.tanh()

    def digest(self) -> str:
        digest = sha256()
        for value in self.parameters():
            digest.update(value.detach().float().cpu().numpy().astype("<f4").tobytes())
        return digest.hexdigest()


@torch.no_grad()
def install_adjustments(endpoint: NativeEndpoint, adjustments: NativeAdjustments, qp: int) -> None:
    if endpoint.order.last is not None:
        raise ValueError("install native weights before the first frame of an endpoint")
    video = endpoint._video()
    endpoint.stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(endpoint.stream):
        for module, gain in ((video.encoder.down, adjustments.encoder_gain()),
                             (video.recon_head.head, adjustments.reconstruction_gain())):
            module.weight.copy_((module.weight.float() * gain[:, None, None, None]).half())
            module.bias.copy_((module.bias.float() * gain).half())
        bias = video.bit_estimator_z.b[qp, :, -1]
        bias.copy_((bias.float() + adjustments.hyper_bias()).half())
        video.update(skip_thres=0)
        video.proxy.set_param(video.add_cdf_to_state_dict(video.state_dict()), video.gaussian_encoder.skip_thres)
    endpoint.stream.synchronize()
