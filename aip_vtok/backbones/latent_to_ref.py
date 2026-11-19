"""Direct 1/16 latent -> 1/8 native I/P reference, with no RGB Decoder call."""

from torch import nn


class DepthwiseResidual(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class LatentToRef(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.width = width
        self.up = nn.Sequential(nn.Conv2d(128, width * 4, 1), nn.PixelShuffle(2))
        self.blocks = nn.Sequential(DepthwiseResidual(width), DepthwiseResidual(width))
        self.intra_head = nn.Conv2d(width, 192, 1)
        self.predictive_head = nn.Conv2d(width, 256, 1)

    def forward(self, latent, intra):
        feature = self.blocks(self.up(latent))
        return self.intra_head(feature) if intra else self.predictive_head(feature)
