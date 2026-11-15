from torch import nn
from torch.nn import functional as F

from .token_grid_projector import TokenGridProjector


class UnifiedPrefixLatentDecoder(nn.Module):
    def __init__(self, dim=192, heads=6, n_max=2048, grid=(17, 30), precision="float64"):
        super().__init__()
        self.projector = TokenGridProjector(dim, heads, n_max, grid, precision)
        self.spatial = nn.Sequential(
            nn.Conv2d(dim, 128, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, padding=1, groups=128),
            nn.GELU(),
        )
        self.outputs = nn.ModuleDict({"I": nn.Conv2d(128, 256, 1), "P": nn.Conv2d(128, 128, 1)})

    def from_grid(self, grid, latent_shape, frame_type):
        feature = self.spatial(grid)
        feature = F.interpolate(feature, size=latent_shape, mode="bilinear", align_corners=False)
        return self.outputs[frame_type](feature)

    def forward(self, embeddings, latent_shape, frame_type, mask=None):
        return self.from_grid(self.projector(embeddings, mask), latent_shape, frame_type)
