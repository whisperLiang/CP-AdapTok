"""The model configuration is also part of the decoder identity."""

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ModelConfig:
    height: int = 1080
    width: int = 1920
    channels: tuple = (48, 64, 96, 128, 192, 192, 192)
    decoder_channels: tuple = (192, 128, 96, 64, 32, 24)
    heads: int = 6
    window: int = 4
    attention_blocks: int = 1
    temporal: bool = True
    codebook_size: int = 8192
    embedding_dim: int = 16
    budgets: tuple = (256, 512, 1024, 2048)
    gop: int = 32
    gradient_checkpointing: bool = False

    def __post_init__(self):
        for name in ("channels", "decoder_channels", "budgets"):
            object.__setattr__(self, name, tuple(getattr(self, name)))
        self.validate_dimensions()
        self.validate_channels()
        self.validate_vq()
        self.validate_budgets()

    def validate_dimensions(self):
        if self.height < 64 or self.width < 64 or self.gop < 1:
            raise ValueError("positive GOP and at least 64x64 input required")
    def validate_channels(self):
        if len(self.channels) != 7 or len(self.decoder_channels) != 6:
            raise ValueError("seven encoder and six decoder channel widths required")
        if min(*self.channels, *self.decoder_channels, self.heads, self.window) < 1:
            raise ValueError("positive channels, heads and window required")
        if any(c % self.heads for c in (*self.channels[4:6], self.decoder_channels[0])):
            raise ValueError("attention widths must be divisible by heads")
        if self.attention_blocks < 0:
            raise ValueError("attention block count must be nonnegative")

    def validate_vq(self):
        if self.codebook_size != 8192 or self.embedding_dim != 16:
            raise ValueError("bitstream v1 requires an 8192x16 codebook")

    def validate_budgets(self):
        if not self.budgets or tuple(sorted(set(self.budgets))) != self.budgets:
            raise ValueError("budgets must be sorted and unique")
        if self.budgets[0] < 1 or self.budgets[-1] > sum(h * w for h, w in self.grid_shapes):
            raise ValueError("budget exceeds available candidates")

    @property
    def padded_height(self):
        return (self.height + 63) // 64 * 64

    @property
    def grid_shapes(self):
        return tuple(((self.padded_height + s - 1) // s, (self.width + s - 1) // s)
                     for s in (128, 64, 32))

    def to_dict(self):
        return asdict(self)
