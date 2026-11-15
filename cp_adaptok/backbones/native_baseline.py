"""Separate upstream CUDA codec control and exact decoded-latent teacher.

No native Proxy is used by CP-AdapTok's new token stream. The native comparison
requires both official extensions, pretrained weights and their supported CUDA ABI.
"""

import io

import torch

from cp_adaptok.backbones.intra_adapter import restore_rgb
from third_party.DCVC.src.models.image_model import DMCI
from third_party.DCVC.src.models.video_model_ld import DMC
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


def load_native_network(cls, path, proxy_cls, device):
    net = cls().eval().to(device).half().to(memory_format=torch.channels_last)
    net.load_state_dict(get_state_dict(path), strict=True)
    net.update(skip_thres=0)
    net.proxy = proxy_cls()
    net.proxy.set_param(net.add_cdf_to_state_dict(net.state_dict()), net.gaussian_encoder.skip_thres)
    return net


def serialize_native(encoded, intra, qp, height, width):
    wire = io.BytesIO()
    sps_bytes = write_sps(wire, {"sps_id": 0, "height": height, "width": width})
    write_ip(wire, intra, 0, qp, encoded["ec_parallel"], False, encoded["bit_stream"])
    return wire.getvalue(), sps_bytes


def native_latent_api_available():
    from inference_extensions_cuda import DMCIProxy, DMCLDProxy

    return all(hasattr(cls, "get_decoded_latent") for cls in (DMCIProxy, DMCLDProxy))


class NativeBaseline:
    def __init__(self, ld_checkpoint, intra_checkpoint, device, qp=32):
        self.qp, self.device = qp, torch.device(device)
        self.custom_stream = torch.cuda.Stream(device=self.device)
        torch.cuda.set_stream(self.custom_stream)
        self.pairs = [self._create_pair(ld_checkpoint, intra_checkpoint) for _ in range(2)]
        self.reset()

    def _create_pair(self, ld_checkpoint, intra_checkpoint):
        from inference_extensions_cuda import DMCIProxy, DMCLDProxy

        specs = ((DMCI, intra_checkpoint, DMCIProxy), (DMC, ld_checkpoint, DMCLDProxy))
        return [load_native_network(cls, path, proxy, self.device) for cls, path, proxy in specs]

    def reset(self):
        self.frame_id = 0
        self.last_decoded_network = None

    def _is_intra(self, force_intra):
        return self.frame_id == 0 or force_intra

    def _encode(self, x, intra, height, width):
        i_enc, p_enc = self.pairs[0]
        if intra:
            return i_enc.compress(x, self.qp, (-height) % 64, (-width) % 64)
        return p_enc.compress(x, self.qp, False, (-height) % 64, (-width) % 64)

    def _decode(self, raw):
        source = io.BytesIO(raw)
        sps_header = read_header(source)
        sps = read_sps_remaining(source, sps_header["sps_id"])
        frame_header = read_header(source)
        qp, ec_part, reset, payload = read_ip_remaining(source)
        i_dec, p_dec = self.pairs[1]
        decoded = (
            i_dec.decompress(payload, sps, qp, ec_part)
            if frame_header["nal_type"] == NalType.NAL_I
            else p_dec.decompress(payload, sps, qp, ec_part, reset)
        )
        self.last_decoded_network = i_dec if frame_header["nal_type"] == NalType.NAL_I else p_dec
        return decoded, payload

    def decoded_latent(self):
        """Return an independent copy of the actual native decoder's quantized input."""
        if self.last_decoded_network is None:
            raise ValueError("decode a native frame before requesting its latent")
        return self.last_decoded_network.proxy.get_decoded_latent()

    def _initialize_reference(self, encoded, decoded):
        self.pairs[0][1].add_ref_feature_from_frame(encoded["x_hat"])
        self.pairs[1][1].add_ref_feature_from_frame(decoded["x_hat"], apply_feature_adaptor=False)

    @torch.no_grad()
    def round_trip(self, rgb, force_intra=False):
        height, width = rgb.shape[-2:]
        intra = self._is_intra(force_intra)
        x = (rgb2ycbcr(rgb) - 0.5).half().to(memory_format=torch.channels_last)
        encoded = self._encode(x, intra, height, width)
        raw, sps_bytes = serialize_native(encoded, intra, self.qp, height, width)
        decoded, payload = self._decode(raw)
        if intra:
            self._initialize_reference(encoded, decoded)
        self.frame_id += 1
        rec = restore_rgb(decoded["x_hat"].float(), height, width)
        counts = {
            "total_bytes": len(raw),
            "payload_bytes": len(payload),
            "sps_bytes": sps_bytes,
            "frame_header_bytes": len(raw) - len(payload) - sps_bytes,
        }
        return rec, counts, raw


class NativeLatentTeacher:
    """Distill y_hat copied from the separate decoder of a real native bitstream.

    The build-only extension patch exposes the very tensor passed to the native
    image/residual decoder. No PyTorch re-encoding or approximate prior is used.
    """

    def __init__(self, ld_checkpoint, intra_checkpoint, device, qp=32):
        if not native_latent_api_available():
            raise RuntimeError("native teacher needs decoded-latent API; run bash scripts/setup_native.sh")
        self.native = NativeBaseline(ld_checkpoint, intra_checkpoint, device, qp)
        self.reset()

    def reset(self):
        self.native.reset()
        self.last_provenance = {}

    @torch.no_grad()
    def target(self, rgb):
        _, counts, _ = self.native.round_trip(rgb)
        y_hat = self.native.decoded_latent().detach().float()
        self.last_provenance = {
            "native_bitstream": True,
            "bytes": counts["total_bytes"],
            "latent_source": "native_decoder_quantized_y_hat",
            "latent_shape": list(y_hat.shape),
        }
        return y_hat
