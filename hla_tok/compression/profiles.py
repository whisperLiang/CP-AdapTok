"""Registered native Compact wire representations."""

from enum import Enum


class Profile(Enum):
    RAW32 = (1, "RAW32", 32, 16)
    Q32_8 = (2, "Q32-8", 32, 8)
    Q32_4 = (3, "Q32-4", 32, 4)
    Q16_8 = (4, "Q16-8", 16, 8)
    Q16_4 = (5, "Q16-4", 16, 4)
    Q8_8 = (6, "Q8-8", 8, 8)
    Q8_6 = (7, "Q8-6", 8, 6)
    Q8_4 = (8, "Q8-4", 8, 4)
    Q4_8 = (9, "Q4-8", 4, 8)
    Q4_4 = (10, "Q4-4", 4, 4)
    PCA32 = (11, "PCA32-BF16", 32, 16)
    PCA24 = (12, "PCA24-BF16", 24, 16)
    PCA16 = (13, "PCA16-BF16", 16, 16)
    PCA8 = (14, "PCA8-BF16", 8, 16)
    PCA4 = (15, "PCA4-BF16", 4, 16)

    def __init__(self, code: int, label: str, dimension: int, bits: int):
        self.code = code
        self.label = label
        self.dimension = dimension
        self.bits = bits

    @property
    def bytes_per_token(self) -> int:
        return self.dimension * self.bits // 8

    @property
    def quantized(self) -> bool:
        return self.bits < 16


PROFILES_BY_CODE = {profile.code: profile for profile in Profile}
PROFILES_BY_LABEL = {profile.label: profile for profile in Profile}
PCA_PROFILES = {profile.dimension: profile for profile in Profile if profile.name.startswith("PCA")}
