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
    read_header,
    read_ip_remaining,
    read_sps_remaining,
    write_ip,
    write_sps,
)
from third_party.DCVC.src.utils.transforms import rgb2ycbcr


class NativeBaseline:
    def __init__(self, ld_checkpoint, intra_checkpoint, device, qp=32):
        from inference_extensions_cuda import DMCIProxy, DMCLDProxy

        self.qp, self.device = qp, torch.device(device)
        self.custom_stream = torch.cuda.Stream(device=self.device)
        torch.cuda.set_stream(self.custom_stream)
        self.pairs = []
        for _ in range(2):
            pair = []
            for cls, path, proxy_cls in (
                (DMCI, intra_checkpoint, DMCIProxy),
                (DMC, ld_checkpoint, DMCLDProxy),
            ):
                net = cls().eval().to(self.device).half().to(memory_format=torch.channels_last)
                net.load_state_dict(get_state_dict(path), strict=True)
                net.update(skip_thres=0)
                net.proxy = proxy_cls()
                net.proxy.set_param(
                    net.add_cdf_to_state_dict(net.state_dict()), net.gaussian_encoder.skip_thres
                )
                pair.append(net)
            self.pairs.append(pair)
        self.reset()

    def reset(self):
        self.frame_id = 0

    @torch.no_grad()
    def round_trip(self, rgb, force_intra=False):
        height, width = rgb.shape[-2:]
        intra = self.frame_id == 0 or force_intra
        x = (rgb2ycbcr(rgb) - 0.5).half().to(memory_format=torch.channels_last)
        i_enc, p_enc = self.pairs[0]
        i_dec, p_dec = self.pairs[1]
        if intra:
            encoded = i_enc.compress(x, self.qp, (-height) % 64, (-width) % 64)
        else:
            encoded = p_enc.compress(x, self.qp, False, (-height) % 64, (-width) % 64)
        sps = {"sps_id": 0, "height": height, "width": width}
        wire = io.BytesIO()
        sps_bytes = write_sps(wire, sps)
        write_ip(wire, intra, 0, self.qp, encoded["ec_parallel"], False, encoded["bit_stream"])
        raw = wire.getvalue()
        # Parse actual serialized metadata rather than sharing encoder-side Python objects.
        source = io.BytesIO(raw)
        sps_header = read_header(source)
        decoded_sps = read_sps_remaining(source, sps_header["sps_id"])
        read_header(source)
        qp, ec_part, reset, payload = read_ip_remaining(source)
        decoded = (
            i_dec.decompress(payload, decoded_sps, qp, ec_part)
            if intra
            else p_dec.decompress(payload, decoded_sps, qp, ec_part, reset)
        )
        if intra:
            p_enc.add_ref_feature_from_frame(encoded["x_hat"])
            p_dec.add_ref_feature_from_frame(decoded["x_hat"], apply_feature_adaptor=False)
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

    @torch.no_grad()
    def target(self, rgb):
        t = self.state.frame_id + 1
        frame_type = "I" if t == 0 else "P"
        ctx = FrozenFrameContext("native-teacher", t, frame_type, *rgb.shape[-2:], self.qp, t - 1, "00" * 16)
        if t:
            ctx.context, ctx.memory = self.ld.begin_frame(self.state)
        adapter = self.intra if t == 0 else self.ld
        net = adapter.model
        y = adapter.encode_latent(prepare_rgb(rgb), ctx)
        if t == 0:
            z_hat = net.hyper_enc(y).round()
            params = net.y_prior_fusion(net.hyper_dec(z_hat))[..., : y.shape[-2], : y.shape[-1]]
            _, _, y_hat, _ = net.forward_prior_4x(
                y,
                adapter.quant("q_scale_y_enc", self.qp),
                adapter.quant("q_scale_y_dec", self.qp),
                params,
                net.y_spatial_prior_reduction,
                net.y_spatial_prior_adaptor_1,
                net.y_spatial_prior_adaptor_2,
                net.y_spatial_prior_adaptor_3,
                net.y_spatial_prior,
            )
        else:
            z_hat = net.hyper_encoder(y).round()
            params = net.res_prior_param_decoder(z_hat, ctx.memory, adapter.quant("q_feature", self.qp))
            _, _, y_hat, _ = net.forward_prior_2x(y, params, net.y_spatial_prior)
        decoded, feature = adapter.decode_latent(y_hat, ctx)
        pytorch_rgb = restore_rgb(decoded, *rgb.shape[-2:])
        native_rgb, counts, _ = self.native.round_trip(rgb)
        error = (pytorch_rgb - native_rgb).abs().max().item()
        if error > self.tolerance:
            raise RuntimeError(f"native teacher parity failed: {error} > {self.tolerance}")
        self.state = ctx.commit(feature, "native").detach()
        self.last_provenance = {
            "native_bitstream": True,
            "bytes": counts["total_bytes"],
            "pixel_max_error": error,
            "tolerance": self.tolerance,
            "latent_source": "official_quantized_prior_verified_against_native_decode",
        }
        return y_hat.detach()
