"""Seven-scale separable convolution backbone with two local attention stages."""

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .window_attention import WindowAttention


def channel_norm(channels):
    groups = next(g for g in (8, 6, 4, 3, 2, 1) if channels % g == 0)
    return nn.GroupNorm(groups, channels)


class SeparableConv(nn.Sequential):
    def __init__(self, inputs, outputs, stride=1):
        super().__init__(nn.Conv2d(inputs, inputs, 3, stride, 1, groups=inputs),
                         nn.Conv2d(inputs, outputs, 1), channel_norm(outputs), nn.SiLU())


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(SeparableConv(channels, channels), nn.Conv2d(channels, channels, 1))

    def forward(self, x):
        return x + self.layers(x) * .5


class CrossScaleFusion(nn.Module):
    """Top-down S128 -> S64 -> S32 fusion before the 16D projections."""

    def __init__(self, channels):
        super().__init__()
        self.projections = nn.ModuleList(nn.Conv2d(c, c, 1) for c in channels)
        self.bridges = nn.ModuleList(nn.Conv2d(a, b, 1) for a, b in zip(channels, channels[1:]))
        self.refine = nn.ModuleList(ResidualBlock(c) for c in channels)

    def forward(self, features):
        outputs = []
        for level, (feature, project, refine) in enumerate(zip(features, self.projections, self.refine)):
            value = project(feature)
            if level:
                coarse = self.bridges[level - 1](outputs[-1])
                value = value + F.interpolate(coarse, size=value.shape[-2:], mode="bilinear", align_corners=False)
            outputs.append(refine(value))
        return tuple(outputs)


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        inputs = 6 if config.temporal else 3
        self.stages = nn.ModuleList()
        for index, channels in enumerate(config.channels):
            blocks = [SeparableConv(inputs, channels, stride=2)]
            if index >= 2:
                blocks.append(ResidualBlock(channels))
            self.stages.append(nn.Sequential(*blocks))
            inputs = channels
        self.attention = nn.ModuleDict({str(i): nn.Sequential(*[
            WindowAttention(config.channels[i], config.heads, config.window)
            for _ in range(config.attention_blocks)]) for i in (4, 5)})
        self.history_fusion = nn.Conv2d(config.channels[4] + config.decoder_channels[0],
                                       config.channels[4], 1) if config.temporal else None
        # Keep common model tensors identically initialized across N0/N1/N2.
        with torch.random.fork_rng(devices=[]):
            self.cross_scale = (CrossScaleFusion(tuple(reversed(config.channels[4:])))
                                if config.encoder_fusion == "cross_scale_v1" else None)

    def pad_input(self, x):
        return F.pad(x, (0, 0, 0, self.config.padded_height - self.config.height), mode="reflect")

    def forward(self, frame, state=None):
        x = frame
        if self.config.temporal:
            previous = torch.zeros_like(frame) if state is None else state.rgb
            x = torch.cat((frame, previous), dim=1)
        x = self.pad_input(x)
        features = []
        for index in range(len(self.stages)):
            x = self.forward_stage(index, x, state)
            if index >= 4:
                features.append(x)
        features = tuple(reversed(features))  # Base, Medium, Fine
        return self.cross_scale(features) if self.cross_scale is not None else features

    def forward_stage(self, index, x, state):
        stage = self.stages[index]
        if self.training and self.config.gradient_checkpointing:
            x = checkpoint(stage, x, use_reentrant=False)
        else:
            x = stage(x)
        if index == 4 and state is not None and self.history_fusion is not None:
            x = x + .25 * self.history_fusion(torch.cat((x, state.feature), dim=1))
        if str(index) in self.attention:
            x = self.attention[str(index)](x)
        return x
