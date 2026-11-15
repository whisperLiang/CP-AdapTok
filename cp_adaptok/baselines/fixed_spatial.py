import math

from torch import nn
from torch.nn import functional as F


class FixedSpatialTokenizer(nn.Module):
    """Row-major pooled spatial control. Only activated by control=fixed_spatial."""

    def __init__(self, n_max, dim):
        super().__init__()
        self.n_max = n_max
        self.input = nn.ModuleDict({"I": nn.Conv2d(256, dim, 1), "P": nn.Conv2d(128, dim, 1)})
        self.history = nn.Conv2d(256, dim, 1)
        self.norm = nn.LayerNorm(dim)

    def forward(self, y, context, frame_type):
        feature = self.input[frame_type](y)
        if context is not None:
            feature = feature + F.adaptive_avg_pool2d(self.history(context), y.shape[-2:])
        height = max(1, round(math.sqrt(self.n_max * y.shape[-2] / y.shape[-1])))
        width = math.ceil(self.n_max / height)
        spatial = F.adaptive_avg_pool2d(feature, (height, width)).flatten(2).transpose(1, 2)
        return self.norm(spatial[:, : self.n_max])
