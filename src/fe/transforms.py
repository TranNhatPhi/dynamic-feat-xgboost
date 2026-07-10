"""
Sáu phép feature engineering (tập Φ) — mỗi phép là một 'arm' cho bandit chọn.

    φ0 identity      giữ nguyên data thô
    φ1 autofeat      mô phỏng vòng đầu Autofeat: 4 phép đơn biến 1/x, x^2, x^3, exp(x)
    φ2 random_proj   Gaussian Random Projection (giảm chiều, bảo toàn khoảng cách - JL)
    φ3 ht_svd        SVD + hard-threshold (Gavish-Donoho) cắt nhiễu trắng
    φ4 robust        RobustScaler (median + IQR, mạnh với outlier)
    φ5 minmax        MinMaxScaler (co giãn về [0,1])
"""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from sklearn.random_projection import GaussianRandomProjection

from src.fe.base import FETransform


class IdentityFE(FETransform):
    name = "identity"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return self._clean(X)


class AutofeatLiteFE(FETransform):
    """Vòng đầu của Autofeat: nối 4 biến đổi đơn biến vào đặc trưng gốc → 5×d chiều.

    Bài gốc giới hạn Autofeat ở vòng đầu với đúng 4 phép: 1/x, x^2, x^3, exp(x)
    để tránh bùng nổ chiều. LƯU Ý: với data nhiều chiều, 5×d có thể tốn RAM."""

    name = "autofeat"

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)   # tránh tràn float32 khi mũ/nghịch đảo
        eps = 1e-8
        # errstate: bỏ qua cảnh báo FP giả của BLAS Apple Accelerate; _clean đảm bảo hữu hạn
        with np.errstate(all="ignore"):
            safe = np.where(np.abs(X) < eps, eps, X)
            inv = 1.0 / safe
            sq = X ** 2
            cube = X ** 3
            ex = np.exp(np.clip(X, -20.0, 20.0))
            out = np.concatenate([X, inv, sq, cube, ex], axis=1)
        return self._clean(out)


class RandomProjectionFE(FETransform):
    """Gaussian Random Projection: giảm chiều bằng ma trận ngẫu nhiên Gaussian."""

    name = "random_proj"

    def __init__(self, ratio: float = 0.5, n_components: int | None = None, seed: int = 0):
        self.ratio = ratio
        self.n_components = n_components
        self.seed = seed

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        d = X.shape[1]
        k = self.n_components or max(1, min(d, max(2, round(self.ratio * d))))
        self.proj_ = GaussianRandomProjection(n_components=k, random_state=self.seed)
        self.proj_.fit(X)
        return self

    def transform(self, X):
        with np.errstate(all="ignore"):  # cảnh báo FP giả của Apple Accelerate
            out = self.proj_.transform(np.asarray(X, dtype=np.float64))
        return self._clean(out)


class HTSVDFE(FETransform):
    """SVD + hard threshold theo ngưỡng tối ưu Gavish-Donoho (nhiễu chưa biết).

    Cắt các giá trị kỳ dị nhỏ (nhiễu trắng), giữ lại thành phần chính → giảm chiều + khử nhiễu."""

    name = "ht_svd"

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        # SVD
        _, s, Vt = np.linalg.svd(Xc, full_matrices=False)
        m, n = Xc.shape
        beta = min(m, n) / max(m, n)
        omega = 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43
        thresh = omega * np.median(s)
        k = int((s > thresh).sum())
        self.k_ = max(1, min(k, Vt.shape[0]))
        self.components_ = Vt[: self.k_]  # (k, d)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=np.float64)
        with np.errstate(all="ignore"):  # cảnh báo FP giả của Apple Accelerate
            out = (X - self.mean_) @ self.components_.T
        return self._clean(out)


class RobustScalerFE(FETransform):
    name = "robust"

    def fit(self, X, y=None):
        self.scaler_ = RobustScaler().fit(X)
        return self

    def transform(self, X):
        return self._clean(self.scaler_.transform(X))


class MinMaxScalerFE(FETransform):
    name = "minmax"

    def fit(self, X, y=None):
        self.scaler_ = MinMaxScaler().fit(X)
        return self

    def transform(self, X):
        return self._clean(self.scaler_.transform(X))
