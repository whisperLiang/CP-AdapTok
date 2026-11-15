import torch
from torch import nn
from torch.nn import functional as F

from cp_adaptok.profiling import measure

from .lightweight_attention import CrossAttentionBlock, spatial_positions


class LearnedQueryTokenizer(nn.Module):
    """Encode all queries once. No budget argument exists at this boundary."""

    def __init__(self, n_max=2048, dim=192, heads=6, layers=2, kv_grid=(17, 30), content_only=False):
        super().__init__()
        if dim % heads or n_max < 1 or layers < 1:
            raise ValueError("invalid query/attention configuration")
        self.n_max, self.dim, self.kv_grid = n_max, dim, tuple(kv_grid)
        self.content_only = content_only
        self.queries = nn.Parameter(torch.randn(1, n_max, dim) * 0.02)
        self.input = nn.ModuleDict({"I": nn.Conv2d(256, dim, 1), "P": nn.Conv2d(128, dim, 1)})
        self.history = nn.Conv2d(256, dim, 1)
        self.high_frequency = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.path_embedding = nn.Parameter(torch.randn(2, 1, dim) * 0.02)
        self.blocks = nn.ModuleList([CrossAttentionBlock(dim, heads) for _ in range(layers)])
        self.norm = nn.LayerNorm(dim)
        self.diagnostics_enabled = False
        self.last_diagnostics = {}

    @torch.no_grad()
    def rescale_initial_queries(self, scale):
        if not self.training or scale <= 0:
            raise ValueError("query rescaling is a positive, training-only initialization")
        self.queries.mul_(scale)

    def record_distribution(self, name, query):
        if self.diagnostics_enabled and self.training:
            from cp_adaptok.diagnostics import distribution_stats

            self.last_diagnostics[name] = distribution_stats(query)

    def spatial_keys(self, y, context, frame_type):
        spatial = self.input[frame_type](y)
        grid = (min(spatial.shape[-2], self.kv_grid[0]), min(spatial.shape[-1], self.kv_grid[1]))
        low = F.adaptive_avg_pool2d(spatial, grid)
        if context is not None:
            low = low + F.adaptive_avg_pool2d(self.history(context), grid)
        detail = self.high_frequency(spatial - F.avg_pool2d(spatial, 3, stride=1, padding=1))
        high = F.adaptive_avg_pool2d(detail, grid)
        pos = spatial_positions(*grid, self.dim, y.device, spatial.dtype)
        keys = torch.cat(
            (
                low.flatten(2).transpose(1, 2) + pos + self.path_embedding[0],
                high.flatten(2).transpose(1, 2) + pos + self.path_embedding[1],
            ),
            dim=1,
        )
        return keys

    def forward(self, y, context, frame_type):
        with measure(self, "encoder.query_feature_projection"):
            keys = self.spatial_keys(y, context, frame_type)
        with measure(self, "encoder.query_cross_attention"):
            return self.read_queries(y, keys)

    def read_queries(self, y, keys):
        query = self.queries.expand(y.shape[0], -1, -1)
        self.record_distribution("initial_query", query)
        for index, block in enumerate(self.blocks):
            query = block(
                query, keys, include_query_residual=not self.content_only or index < len(self.blocks) - 1
            )
            self.record_distribution(f"block_{index}", query)
        output = self.norm(query)
        self.record_distribution("output", output)
        return output
