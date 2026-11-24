"""Experimental four-frame KV replay of the pinned official AdapTok model."""

from dataclasses import dataclass, field

import torch
from torch.nn import functional as F

from experiments.original_adaptok_1080p.tiling import Stitcher, TileLayout, extract_batches
from hla_tok.evaluation.io import timed_cuda, write_csv, write_json
from hla_tok.evaluation.official import wire_indices


@dataclass
class KVHistory:
    positions: torch.Tensor | None = None
    keys: list = field(default_factory=list)
    values: list = field(default_factory=list)

    def forward(self, current, indices, blocks, mask):
        order = indices if self.positions is None else torch.cat((self.positions, indices))
        # Match the official order of image and latent slots when attending to the cached history.
        permutation = order.argsort()
        allowed = mask.index_select(0, indices).index_select(1, order[permutation])[None, None].contiguous()
        fresh_keys, fresh_values = [], []
        for index, block in enumerate(blocks):
            attention = block.attn
            b, n, c = current.shape
            projected = attention.qkv(block.norm1(current))
            q, k, v = projected.reshape(b, n, 3, attention.num_heads, attention.head_dim).permute(2, 0, 3, 1, 4)
            q, k = attention.q_norm(q), attention.k_norm(k)
            if self.positions is not None:
                k = torch.cat((self.keys[index], k), dim=2)
                v = torch.cat((self.values[index], v), dim=2)
            fresh_keys.append(k)
            fresh_values.append(v)
            mixed = F.scaled_dot_product_attention(q, k[:, :, permutation], v[:, :, permutation], attn_mask=allowed)
            mixed = attention.proj_drop(attention.proj(mixed.transpose(1, 2).reshape(b, n, c)))
            current = current + block.drop_path1(block.ls1(mixed))
            current = current + block.drop_path2(block.ls2(block.mlp(block.norm2(current))))
        self.positions, self.keys, self.values = order, fresh_keys, fresh_values
        return current


@dataclass
class FourFrameWindow:
    baseline: object
    encoding: KVHistory = field(default_factory=KVHistory)
    decoding: KVHistory = field(default_factory=KVHistory)

    @staticmethod
    def _group_indices(group, width, device, offset=0, keep=None):
        start = group * width + offset
        return torch.arange(start, start + (width if keep is None else keep), device=device)

    def encode(self, frames, group):
        model = self.baseline.model
        patches = model.x_embedder
        pos = self._group_indices(group, patches.num_spatial_patches, frames.device)
        ranks = self._group_indices(group, self.baseline.ranks, frames.device)
        x = patches.norm(patches.proj(frames).flatten(2).transpose(1, 2)) + model.get_encoder_patch_pe()[..., pos, :]
        q = model.get_encoder_latent_query_embed()[..., ranks, :].expand(len(frames), -1, -1)
        out = self.encoding.forward(torch.cat((x, q), dim=1), torch.cat((pos, model.video_token_num + ranks)),
                                    model.encoder.blocks, model.encoder_attn_mask)
        return model.bottleneck(out[:, -self.baseline.ranks:])["bottleneck_rep"]

    def decode(self, ids, group):
        model = self.baseline.model
        budget = ids.shape[-1]
        ranks = self._group_indices(group, self.baseline.ranks, ids.device, keep=budget)
        pos = self._group_indices(group, model.x_embedder.num_spatial_patches, ids.device)
        q = model.get_decoder_patch_query_embed()[..., pos, :].expand(len(ids), -1, -1)
        z = model.bottleneck.decode(ids) + model.get_decoder_latent_pe()[..., ranks, :]
        out = self.decoding.forward(torch.cat((z, q), dim=1), torch.cat((ranks, model.bottleneck_token_num + pos)),
                                    model.decoder.blocks, model.decoder_attn_mask)
        return model.unpatchify(model.final_layer(out[:, budget:])).float().clamp(0, 1)


def compare_tile(baseline, tiles, budget):
    source_ids, _ = baseline.encode(tiles)
    direct = baseline.decode(source_ids[..., :budget], budget)
    cache = FourFrameWindow(baseline)
    ordered, decoded = [], []
    for group in range(4):
        ids = cache.encode(tiles[:, :, group*4:(group+1)*4], group)
        ordered.append(ids)
        decoded.append(cache.decode(ids[:, :budget], group))
    delta = (torch.cat(decoded, dim=2) - direct).abs()
    return {"integer_ids_equal": torch.equal(torch.stack(ordered, dim=1), source_ids),
            "rgb_max_abs": delta.max().item(), "rgb_mean_abs": delta.mean().item()}


def reconstruct_group(clip, group, layout, baseline, cached, batch, budget):
    sink = Stitcher(layout, 4, 3, clip.device)
    for positions, tiles in extract_batches(clip, layout, batch):
        if positions not in cached:
            cached[positions] = FourFrameWindow(baseline)
        node = cached[positions]
        ids = node.encode(tiles, group)
        _, received = wire_indices(ids, budget)
        sink.add(positions, node.decode(received, group))
    return sink.finish()


@torch.inference_mode()
def run_cache_diagnostics(clip, baseline, output):
    layout = TileLayout(1080, 1920)
    source = clip.cuda().float().div_(255)
    tiles = next(extract_batches(source, layout, 1))[1]
    rows, comparisons = [], {}
    for precision in ("fp32", "bf16"):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
            comparisons[precision] = {str(k): compare_tile(baseline, tiles, k) for k in (128, 256, 512)}
            write_json(output / "cache_parity.json", comparisons)
            # Record experimental latency even if parity fails. The official baseline remains eager.
            cached = {}
            torch.cuda.reset_peak_memory_stats()
            for group in range(4):
                _, wall, cuda = timed_cuda(lambda: reconstruct_group(
                    source[group*4:(group+1)*4], group, layout, baseline, cached, 4, 256
                ))
                rows.append({"precision": precision, "group": group, "source_frames": 4, "tile_batch": 4,
                             "wall_ms": wall, "cuda_ms": cuda, "codec_fps": 4000 / wall,
                             "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20})
                write_csv(output / "four_frame_cache_latency.csv", rows)
            cached.clear()
    return comparisons
