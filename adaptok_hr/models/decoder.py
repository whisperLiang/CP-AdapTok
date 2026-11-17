"""Decode only received embeddings and receiver-owned history."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .encoder import ResidualBlock, SeparableConv
from .window_attention import WindowAttention


class Upsample(nn.Module):
    def __init__(self, inputs, outputs):
        super().__init__()
        self.project = nn.Conv2d(inputs, outputs, 1)
        self.refine = SeparableConv(outputs, outputs)

    def forward(self, x, shape):
        return self.refine(F.interpolate(self.project(x), size=shape, mode="bilinear", align_corners=False))


class Decoder(nn.Module):
    def __init__(self, config, table):
        super().__init__()
        self.config = config
        object.__setattr__(self, "table", table)
        low = config.decoder_channels[0]
        self.restore = nn.ModuleList(nn.Conv2d(config.embedding_dim + 1, low, 1) for _ in range(3))
        self.fusion = nn.ModuleList(ResidualBlock(low) for _ in range(3))
        self.attention = nn.ModuleList(nn.Sequential(*[
            WindowAttention(low, config.heads, config.window) for _ in range(config.attention_blocks)])
            for _ in range(2))
        self.temporal_fusion = nn.Conv2d(low * 2, low, 1) if config.temporal else None
        self.rgb_history_logit = nn.Parameter(torch.tensor(-2.0)) if config.temporal else None
        widths = config.decoder_channels
        self.upsample = nn.ModuleList(Upsample(a, b) for a, b in zip(widths, widths[1:]))
        self.rgb_head = nn.Sequential(nn.Conv2d(widths[-1], 12, 1), nn.PixelShuffle(2))

    def restore_tokens(self, tokens, state=None):
        return self.fuse_maps(self.table.scatter(tokens), state)

    def fuse_maps(self, maps, state=None):
        x = self.fusion[0](self.restore[0](maps[0]))
        for level in (1, 2):
            x = F.interpolate(x, size=maps[level].shape[-2:], mode="bilinear", align_corners=False)
            x = self.fusion[level](x + self.restore[level](maps[level]))
            x = self.attention[level - 1](x)
        if state is not None and self.temporal_fusion is not None:
            x = x + .25 * self.temporal_fusion(torch.cat((x, state.feature), dim=1))
        return x

    def upsample_rgb(self, feature, state=None):
        x = feature
        # First mix channels at S32, then use four doublings to S2 and pixel shuffle to RGB.
        for index, layer in enumerate(self.upsample):
            scale = 32 if index == 0 else 32 // (2**index)
            shape = ((self.config.padded_height + scale - 1) // scale,
                     (self.config.width + scale - 1) // scale)
            if self.training and self.config.gradient_checkpointing:
                x = checkpoint(layer, x, shape, use_reentrant=False)
            else:
                x = layer(x, shape)
        rgb = self.rgb_head(x).sigmoid()[:, :, :self.config.height, :self.config.width]
        if state is not None and self.rgb_history_logit is not None:
            weight = .5 * self.rgb_history_logit.sigmoid()
            rgb = (1 - weight) * rgb + weight * state.rgb
        return rgb

    def forward(self, tokens, state=None):
        feature = self.restore_tokens(tokens, state)
        return self.upsample_rgb(feature, state), feature
