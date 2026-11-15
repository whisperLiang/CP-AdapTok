"""Separate upstream CUDA codec control and parity-checked latent teacher.

No native Proxy is used by CP-AdapTok's new token stream. The native comparison
requires both official extensions, pretrained weights and their supported CUDA ABI.
"""

import io

import torch

from cp_adaptok.backbones.causal_state import CausalState, FrozenFrameContext
from cp_adaptok.backbones.intra_adapter import IntraBackboneAdapter, prepare_rgb, restore_rgb
from cp_adaptok.backbones.ld_adapter import CausalLDBackboneAdapter
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
        return decoded, payload

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
    """Quantized y_hat is supervised only after parity with real native decoding.

    The upstream Proxy does not export its internal y_hat. We reconstruct that
    quantized representation via its official PyTorch prior, require pixel parity
    against a real separately decoded native stream, and report the error. A failed
    parity check is fatal, never replaced by an unquantized encoder latent.
    """

    def __init__(self, ld_checkpoint, intra_checkpoint, device, qp=32, tolerance=0.02):
        self.native = NativeBaseline(ld_checkpoint, intra_checkpoint, device, qp)
        self.ld = CausalLDBackboneAdapter(ld_checkpoint).to(device).eval().requires_grad_(False)
        self.intra = IntraBackboneAdapter(intra_checkpoint).to(device).eval().requires_grad_(False)
        self.qp, self.tolerance = qp, tolerance
        self.reset()

    def reset(self):
        self.native.reset()
        self.state = CausalState.empty("native-teacher")
        self.last_provenance = {}

    def _frame_context(self, rgb):
        t = self.state.frame_id + 1
        frame_type = "I" if t == 0 else "P"
        ctx = FrozenFrameContext("native-teacher", t, frame_type, *rgb.shape[-2:], self.qp, t - 1, "00" * 16)
        if t:
            ctx.context, ctx.memory = self.ld.begin_frame(self.state)
        return ctx

    def _intra_quantized(self, y):
        net = self.intra.model
        z_hat = net.hyper_enc(y).round()
        params = net.y_prior_fusion(net.hyper_dec(z_hat))[..., : y.shape[-2], : y.shape[-1]]
        _, _, y_hat, _ = net.forward_prior_4x(
            y,
            self.intra.quant("q_scale_y_enc", self.qp),
            self.intra.quant("q_scale_y_dec", self.qp),
            params,
            net.y_spatial_prior_reduction,
            net.y_spatial_prior_adaptor_1,
            net.y_spatial_prior_adaptor_2,
            net.y_spatial_prior_adaptor_3,
            net.y_spatial_prior,
        )
        return y_hat

    def _inter_quantized(self, y, ctx):
        net = self.ld.model
        z_hat = net.hyper_encoder(y).round()
        params = net.res_prior_param_decoder(z_hat, ctx.memory, self.ld.quant("q_feature", self.qp))
        _, _, y_hat, _ = net.forward_prior_2x(y, params, net.y_spatial_prior)
        return y_hat

    def _verify_native(self, pytorch_rgb, rgb):
        native_rgb, counts, _ = self.native.round_trip(rgb)
        error = (pytorch_rgb - native_rgb).abs().max().item()
        if error > self.tolerance:
            raise RuntimeError(f"native teacher parity failed: {error} > {self.tolerance}")
        self.last_provenance = {
            "native_bitstream": True,
            "bytes": counts["total_bytes"],
            "pixel_max_error": error,
            "tolerance": self.tolerance,
            "latent_source": "official_quantized_prior_verified_against_native_decode",
        }

    @torch.no_grad()
    def target(self, rgb):
        ctx = self._frame_context(rgb)
        adapter = self.intra if ctx.frame_type == "I" else self.ld
        y = adapter.encode_latent(prepare_rgb(rgb), ctx)
        y_hat = self._intra_quantized(y) if ctx.frame_type == "I" else self._inter_quantized(y, ctx)
        decoded, feature = adapter.decode_latent(y_hat, ctx)
        self._verify_native(restore_rgb(decoded, *rgb.shape[-2:]), rgb)
        self.state = ctx.commit(feature, "native").detach()
        return y_hat.detach()
