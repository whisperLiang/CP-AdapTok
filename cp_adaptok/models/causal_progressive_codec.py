import hashlib
import json
from dataclasses import asdict, dataclass

import torch
from torch import nn

from cp_adaptok.backbones.causal_state import CausalState, FrozenFrameContext
from cp_adaptok.backbones.intra_adapter import IntraBackboneAdapter, prepare_rgb, restore_rgb
from cp_adaptok.backbones.ld_adapter import CausalLDBackboneAdapter
from cp_adaptok.decoder.prefix_latent_decoder import UnifiedPrefixLatentDecoder
from cp_adaptok.profiling import measure
from cp_adaptok.streaming.causal_scorer import CausalScorer
from cp_adaptok.streaming.entropy_codec import BlockEntropyModel
from cp_adaptok.tokenizer.learned_queries import LearnedQueryTokenizer
from cp_adaptok.tokenizer.vector_quantizer import AdapTokVQBottleneck


@dataclass(frozen=True)
class CodecConfig:
    n_max: int = 2048
    k_ref: int = 128
    packet_tokens: int = 128
    dim: int = 192
    heads: int = 6
    layers: int = 2
    codebook_size: int = 8192
    embedding_dim: int = 16
    grid: tuple = (17, 30)
    kv_grid: tuple = (17, 30)
    qp: int = 32
    attention_precision: str = "float64"
    state_fingerprint: str = "mixed_int64_v1"
    query_content_only: bool = False

    def __post_init__(self):
        object.__setattr__(self, "grid", tuple(self.grid))
        object.__setattr__(self, "kv_grid", tuple(self.kv_grid))
        self._validate_queries()
        self._validate_stream_layout()
        self._validate_formats()

    def _validate_queries(self):
        if min(self.dim, self.heads, self.layers, self.embedding_dim) <= 0 or self.dim % self.heads:
            raise ValueError("invalid attention dimensions")
        if len(self.grid) != 2 or len(self.kv_grid) != 2:
            raise ValueError("grids must have two dimensions")
        if min(*self.grid, *self.kv_grid) < 1:
            raise ValueError("grid dimensions must be positive")

    def _validate_stream_layout(self):
        if not (1 <= self.k_ref <= self.n_max <= 8192):
            raise ValueError("invalid model configuration")
        if self.packet_tokens not in (8, 16, 32, 64, 128, 256):
            raise ValueError("invalid packet size")
        if self.n_max % self.packet_tokens or self.k_ref % self.packet_tokens:
            raise ValueError("reference and maximum budgets must be packet aligned")

    def _validate_formats(self):
        if not 2 <= self.codebook_size <= 32768 or not 0 <= self.qp < 64:
            raise ValueError("invalid codebook/QP")
        if self.attention_precision not in ("float32", "float64"):
            raise ValueError("invalid attention accumulation precision")
        if self.state_fingerprint != "mixed_int64_v1":
            raise ValueError("unsupported state fingerprint algorithm")


@dataclass
class EncodedFrame:
    context: FrozenFrameContext
    features: torch.Tensor
    vq: object

    def prefix(self, k):
        if not 1 <= k <= self.vq.ids.shape[1]:
            raise ValueError("invalid prefix")
        return self.vq.ids[:, :k]


class CausalProgressiveCodec(nn.Module):
    def __init__(self, config=None, ld_checkpoint=None, intra_checkpoint=None):
        super().__init__()
        self.config = config or CodecConfig()
        c = self.config
        self.ld = CausalLDBackboneAdapter(ld_checkpoint)
        self.intra = IntraBackboneAdapter(intra_checkpoint)
        self.tokenizer = LearnedQueryTokenizer(
            c.n_max, c.dim, c.heads, c.layers, c.kv_grid, c.query_content_only
        )
        self.vq = AdapTokVQBottleneck(c.dim, c.codebook_size, c.embedding_dim)
        self.decoder = UnifiedPrefixLatentDecoder(c.dim, c.heads, c.n_max, c.grid, c.attention_precision)
        self.entropy_model = BlockEntropyModel(c.codebook_size, c.n_max // c.packet_tokens)
        self.scorer = CausalScorer(c.dim)
        self._model_id = None
        self.freeze_backbones()

    def freeze_backbones(self):
        self.ld.requires_grad_(False)
        self.intra.requires_grad_(False)

    @property
    def device(self):
        return self.vq.codebook.weight.device

    @property
    def model_id(self):
        if self._model_id is None:
            identity_config = asdict(self.config)
            if not self.config.query_content_only:
                identity_config.pop("query_content_only")
            digest = hashlib.sha256(json.dumps(identity_config, sort_keys=True).encode())
            for name, tensor in sorted(self.state_dict().items()):
                digest.update(name.encode())
                digest.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
            self._model_id = digest.hexdigest()[:32]
        return self._model_id

    def refresh_identity(self):
        self._model_id = None
        return self.model_id

    def _validate_frame_request(self, state, frame_id, frame_type, height, width, qp):
        if state.model_id != self.model_id:
            raise ValueError("model_id mismatch")
        if frame_type not in ("I", "P") or frame_id < 0:
            raise ValueError("invalid frame type/id")
        if not 0 <= qp < 64 or min(height, width) < 1 or max(height, width) > 8192:
            raise ValueError("invalid shape/QP")

    def begin_frame(self, state, frame_id, frame_type, height, width, qp=None):
        qp = self.config.qp if qp is None else qp
        self._validate_frame_request(state, frame_id, frame_type, height, width, qp)
        if frame_type == "I":
            return FrozenFrameContext(self.model_id, frame_id, "I", height, width, qp, -1, "00" * 16)
        if frame_id != state.frame_id + 1:
            raise ValueError("P frame reference_id mismatch")
        context, memory = self.ld.begin_frame(state)
        expected = ((height + 63) // 64 * 8, (width + 63) // 64 * 8)
        if context.shape[-2:] != expected:
            raise ValueError("resolution change requires an I frame reset")
        return FrozenFrameContext(
            self.model_id,
            frame_id,
            "P",
            height,
            width,
            qp,
            state.frame_id,
            state.fingerprint(),
            context,
            memory,
        )

    def encode_full(self, rgb, state, frame_id=None, force_intra=False):
        if rgb.shape[0] != 1:
            raise ValueError("stream model uses B=1; batch independent sequences outside the state machine")
        frame_id = state.frame_id + 1 if frame_id is None else frame_id
        frame_type = "I" if state.feature is None or force_intra else "P"
        with measure(self, "encoder.reference_context"):
            context = self.begin_frame(state, frame_id, frame_type, *rgb.shape[-2:])
        adapter = self.intra if frame_type == "I" else self.ld
        with measure(self, "encoder.preprocess"):
            prepared = prepare_rgb(rgb)
        with measure(self, "encoder.ld_dmci"):
            y = adapter.encode_latent(prepared, context)
        with measure(self, "encoder.learned_queries"):
            self.tokenizer.pipeline_profiler = getattr(self, "pipeline_profiler", None)
            features = self.tokenizer(y, context.context, frame_type)
        with measure(self, "encoder.vq_search"):
            quantized = self.vq(features)
        return EncodedFrame(context, features, quantized)

    def decode_embeddings(self, embeddings, context, mask=None):
        with measure(self, "decoder.token_to_latent"):
            y_hat = self.decoder(embeddings, context.latent_shape, context.frame_type, mask)
        return self.decode_latent(y_hat, context)

    def decode_latent(self, y_hat, context):
        adapter = self.intra if context.frame_type == "I" else self.ld
        with measure(self, "decoder.ld_dmci"):
            yuv, feature = adapter.decode_latent(y_hat, context)
        with measure(self, "decoder.postprocess"):
            rgb = restore_rgb(yuv, context.height, context.width)
        return rgb, feature

    def decode_ids(self, ids, context):
        with measure(self, "decoder.embedding_lookup"):
            embeddings = self.vq.embed(ids)
        return self.decode_embeddings(embeddings, context)

    def empty_state(self):
        return CausalState.empty(self.model_id)
