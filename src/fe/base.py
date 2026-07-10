"""Lớp cơ sở cho mọi phép feature engineering (mỗi phép = một 'arm' của bandit)."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class FETransform(ABC):
    """
    Giao diện chung: fit trên train, transform áp cho mọi tập.

    QUAN TRỌNG chống rò rỉ (leakage): mọi tham số học được (ma trận chiếu, thành phần
    SVD, median/IQR của scaler...) phải fit CHỈ trên train, rồi áp y hệt cho val/test.
    Trong vòng boosting, transform còn được lưu lại để áp cho test lúc dự đoán.
    """

    name: str = "base"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "FETransform":
        ...

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        ...

    def fit_transform(self, X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    # Biên an toàn nằm trong dải float32 (max ~3.4e38) để ép kiểu không hoá inf.
    _CLIP = 1e30

    @classmethod
    def _clean(cls, X: np.ndarray) -> np.ndarray:
        """Bảo vệ downstream: tính ở float64, kẹp biên, ép float32 liên tục.

        Giữ dấu của giá trị tràn (posinf->+CLIP) thay vì gộp về 0 (bảo toàn thứ tự
        cho cây quyết định) — tránh âm thầm mất thông tin."""
        X = np.asarray(X, dtype=np.float64)
        X = np.nan_to_num(X, nan=0.0, posinf=cls._CLIP, neginf=-cls._CLIP)
        X = np.clip(X, -cls._CLIP, cls._CLIP)
        return np.ascontiguousarray(X, dtype=np.float32)
