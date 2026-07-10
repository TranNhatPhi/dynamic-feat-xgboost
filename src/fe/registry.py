"""Đăng ký các phép FE — thứ tự này = thứ tự 'arm' của bandit (giữ cố định)."""

from __future__ import annotations

from src.fe.base import FETransform
from src.fe.transforms import (
    AutofeatLiteFE,
    HTSVDFE,
    IdentityFE,
    MinMaxScalerFE,
    RandomProjectionFE,
    RobustScalerFE,
)

# Thứ tự cố định — index dùng làm 'arm id' cho bandit.
FE_FACTORIES = {
    "identity": lambda seed=0: IdentityFE(),
    "autofeat": lambda seed=0: AutofeatLiteFE(),
    "random_proj": lambda seed=0: RandomProjectionFE(seed=seed),
    "ht_svd": lambda seed=0: HTSVDFE(),
    "robust": lambda seed=0: RobustScalerFE(),
    "minmax": lambda seed=0: MinMaxScalerFE(),
}

FE_ARMS: list[str] = list(FE_FACTORIES)  # ['identity','autofeat',...]


def make_fe(name: str, seed: int = 0) -> FETransform:
    if name not in FE_FACTORIES:
        raise ValueError(f"Phép FE không rõ: {name}. Có: {FE_ARMS}")
    return FE_FACTORIES[name](seed=seed)


# Các tập con phục vụ ablation (xem docs/03)
FE_CHEAP = "random_proj"       # phép rẻ nhất cho cấu hình A (fixed-cheap)
FE_ALL = FE_ARMS               # cấu hình C (all-FE, như bài gốc)
