"""Install fitted official parameters on both HUF1 peers before coding a frame."""

from hashlib import sha256

import torch

from hla_tok.uf_v2.runtime import Unified

NATIVE_TRAINABLE = {
    "native_i": ("enc.enc_2.6.", "dec.dec_1.12.", "dec.dec_2."),
    "native_p": ("encoder.", "decoder.", "recon_head.head.", "feature_extractor.conv.4.",
                 "hyper_encoder.conv.2.", "bit_estimator_z."),
}


def native_parameters(network, section):
    prefixes = NATIVE_TRAINABLE[section]
    return {name: param for name, param in network.named_parameters() if name.startswith(prefixes)}


def inference_digest(overlays):
    state = sha256(b"HUF1 native parameter overlay\x00")
    for kind, weights in sorted(overlays.items()):
        for name, parameter in sorted(weights.items()):
            state.update(f"{kind}/{name}/{tuple(parameter.shape)}\0".encode())
            state.update(parameter.detach().cpu().half().contiguous().numpy().astype("<f2", copy=False).tobytes())
    return state.digest()


class JointUnified(Unified):
    def __init__(self, *args, overlays, **kwargs):
        if set(overlays) != set(NATIVE_TRAINABLE):
            raise ValueError("both official I and P overlay sections are required")
        self.overlays = overlays
        super().__init__(*args, **kwargs)
        self.tag = sha256(self.tag + inference_digest(overlays)).digest()[:8]

    @torch.no_grad()
    def _install(self, endpoint):
        super()._install(endpoint)
        for kind, network in (("native_i", endpoint.intra), ("native_p", endpoint._video())):
            expected = native_parameters(network, kind)
            fitted = self.overlays[kind]
            if fitted.keys() != expected.keys():
                raise ValueError(f"incomplete or unknown parameters in {kind}")
            if any(fitted[key].shape != tensor.shape or not torch.isfinite(fitted[key]).all()
                   for key, tensor in expected.items()):
                raise ValueError(f"invalid parameter tensor in {kind}")
            with torch.cuda.stream(endpoint.stream):
                for key, value in fitted.items():
                    expected[key].copy_(value)
                network.update(skip_thres=0)
                network.proxy.set_param(network.add_cdf_to_state_dict(network.state_dict()),
                                        network.gaussian_encoder.skip_thres)
        endpoint.stream.synchronize()
