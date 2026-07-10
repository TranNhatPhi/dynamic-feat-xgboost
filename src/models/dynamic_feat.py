"""
NOVELTY — Dynamic Feat-XGBoost: thay lịch xoay vòng cứng (i%6) bằng contextual bandit.

Chỉ cần một FESelector mới bọc LinUCB; lõi FeatXGBoost giữ nguyên. Đây là toàn bộ
"điểm cắm" của đóng góp mới — không phá cấu trúc baseline.
"""

from __future__ import annotations

import numpy as np

from src.models.bandit import LinUCB
from src.models.feat_xgboost import CONTEXT_DIM, FESelector


class BanditSelector(FESelector):
    """Bộ chọn FE học được: LinUCB nhìn context (đặc trưng tập mẫu khó) -> chọn 1 phép FE."""

    uses_reward = True

    def __init__(self, fe_names: list[str], alpha: float = 1.0, seed: int = 0):
        self.bandit = LinUCB(fe_names, context_dim=CONTEXT_DIM, alpha=alpha, seed=seed)

    def choose(self, step: int, context: np.ndarray) -> int:
        return self.bandit.select(context)

    def reward(self, feat_id: int, reward: float, context: np.ndarray) -> None:
        self.bandit.update(feat_id, reward, context)

    def selection_frequency(self):
        """Tần suất chọn mỗi phép FE — dữ liệu cho phần 'interpretable policy' của paper."""
        return self.bandit.selection_frequency()
