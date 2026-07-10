"""
Lõi Feat-XGBoost — tái hiện Algorithm 2 của bài gốc (bffeatxgbAv51), có HOOK để cắm bandit.

Điểm bám code gốc:
  - Precompute 6 phiên bản FE của toàn tập; mỗi bước chỉ index theo GOSS.
  - GOSS a=0.4, b=0.3 (theo code gốc).
  - Weak learner = 1 vòng boosting (xgb.train num_boost_round=1), objective multi:softprob.
  - Mỗi weak learner τ_l ghép cố định với phép FE của bước đó -> lúc predict dùng lại đúng không gian.

Khác code gốc (có chủ đích, an toàn hơn):
  - Dùng API thấp `xgboost.train` + DMatrix để KHOÁ num_class=K (tránh lỗi khi tập mẫu
    khó thiếu lớp — sklearn XGBClassifier tự suy luận num_class sẽ sai shape).
  - Bộ chọn FE tách thành FESelector (baseline & dynamic dùng chung lõi này).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import xgboost as xgb

from src.fe.registry import FE_ARMS, make_fe
from src.goss import goss_sample

CONTEXT_DIM = 6  # [bias, mean|g|, std|g|, class_coverage, class_entropy, n_feat_norm]


# --------------------------------------------------------------------------- #
# Bộ chọn FE — HOOK. Baseline & novelty đều là một FESelector.
# --------------------------------------------------------------------------- #
class FESelector(ABC):
    uses_reward: bool = False

    @abstractmethod
    def choose(self, step: int, context: np.ndarray) -> int:
        ...

    def reward(self, feat_id: int, reward: float, context: np.ndarray) -> None:
        pass  # no-op cho baseline


class RoundRobinSelector(FESelector):
    """Baseline BÀI GỐC: feat_id = step % n."""

    def __init__(self, n_arms: int):
        self.n = n_arms

    def choose(self, step, context):
        return step % self.n


class FixedSelector(FESelector):
    """Ablation A: luôn dùng 1 phép FE."""

    def __init__(self, arm: int):
        self.arm = arm

    def choose(self, step, context):
        return self.arm


class RandomSelector(FESelector):
    """Ablation B: chọn ngẫu nhiên 1 phép FE mỗi bước."""

    def __init__(self, n_arms: int, seed: int = 0):
        self.n = n_arms
        self.rng = np.random.default_rng(seed)

    def choose(self, step, context):
        return int(self.rng.integers(self.n))


# --------------------------------------------------------------------------- #
def _softmax(F: np.ndarray) -> np.ndarray:
    F = F - F.max(axis=1, keepdims=True)
    e = np.exp(F)
    return e / e.sum(axis=1, keepdims=True)


def _logloss(proba: np.ndarray, y: np.ndarray) -> float:
    p = np.clip(proba[np.arange(len(y)), y], 1e-12, 1.0)
    return float(-np.log(p).mean())


class FeatXGBoost:
    def __init__(
        self,
        selector: FESelector | None = None,
        n_boost: int = 100,
        a: float = 0.4,
        b: float = 0.3,
        fe_names: list[str] = FE_ARMS,
        xgb_cfg: dict | None = None,
        lam: float = 0.01,       # hệ số time-penalty: Reward = Δacc − lam·cost(FE)
        warmup: int = 10,        # số bước đầu ép round-robin cho bandit "nếm" đủ arm
        seed: int = 0,
    ):
        self.selector = selector or RoundRobinSelector(len(fe_names))
        self.n_boost = n_boost
        self.a, self.b = a, b
        self.fe_names = list(fe_names)
        self.xgb_cfg = xgb_cfg or {"eta": 0.3, "max_depth": 6}
        self.lam = lam
        self.warmup = warmup
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # ---- precompute FE cho toàn tập (fit trên train) ---------------------- #
    def _precompute(self, Xtr, ytr, Xval, Xte):
        self.Z = {"train": {}, "val": {}, "test": {}}
        self._dmat = {"train": {}, "val": {}, "test": {}}
        for name in self.fe_names:
            fe = make_fe(name, seed=self.seed)
            fe.fit(Xtr, ytr)
            self.Z["train"][name] = fe.transform(Xtr)
            if Xval is not None:
                self.Z["val"][name] = fe.transform(Xval)
            if Xte is not None:
                self.Z["test"][name] = fe.transform(Xte)
        # DMatrix đầy đủ (không base_margin) để cập nhật/ dự đoán — cache cho nhanh
        for which in ("train", "val", "test"):
            for name, Z in self.Z[which].items():
                self._dmat[which][name] = xgb.DMatrix(Z)
        # chi phí mỗi phép FE = số chiều đầu ra / số chiều gốc (proxy cho chi phí dựng cây)
        d0 = Xtr.shape[1]
        self.fe_cost = np.array(
            [self.Z["train"][name].shape[1] / d0 for name in self.fe_names],
            dtype=float,
        )

    # ---- context cho bandit (rẻ) ----------------------------------------- #
    def _context(self, idx, gnorm, n_feat) -> np.ndarray:
        gi = gnorm[idx]
        yi = self._ytr[idx]
        counts = np.bincount(yi, minlength=self.K).astype(float)
        coverage = float((counts > 0).sum()) / self.K
        p = counts / max(counts.sum(), 1)
        p = p[p > 0]
        entropy = float(-(p * np.log(p)).sum() / np.log(self.K)) if self.K > 1 else 0.0
        return np.array([
            1.0,                              # bias
            float(gi.mean()),                 # độ khó trung bình
            float(gi.std()),                  # độ phân tán gradient
            coverage,                         # tỉ lệ lớp có mặt
            entropy,                          # entropy phân bố lớp
            min(n_feat / 256.0, 1.0),         # số chiều gốc (chuẩn hoá)
        ], dtype=float)

    # ---- huấn luyện ------------------------------------------------------- #
    def fit(self, Xtr, ytr, Xval=None, yval=None, Xte=None):
        self._ytr = np.asarray(ytr).astype(int).ravel()
        self.K = int(self._ytr.max() + 1)
        N = Xtr.shape[0]
        n_feat = Xtr.shape[1]
        self._precompute(Xtr, ytr, Xval, Xte)

        onehot = np.eye(self.K)[self._ytr]
        Fm = np.zeros((N, self.K))                     # margin tích luỹ (train)
        track_reward = self.selector.uses_reward and Xval is not None
        if track_reward:
            yval_i = np.asarray(yval).astype(int).ravel()
            Fm_val = np.zeros((Xval.shape[0], self.K))
            prev_val_acc = (np.argmax(Fm_val, axis=1) == yval_i).mean()

        base_params = {
            "objective": "multi:softprob",
            "num_class": self.K,
            "verbosity": 0,
            "seed": self.seed,
            **self.xgb_cfg,
        }
        n_arms = len(self.fe_names)
        self.models: list[tuple[xgb.Booster, int]] = []
        self.feat_history: list[int] = []              # nguồn sự thật cho Hình 2
        for step in range(self.n_boost):
            p = _softmax(Fm)
            g = p - onehot                              # gradient bậc 1 (N×K)
            gnorm = np.linalg.norm(g, axis=1)

            idx, w = goss_sample(gnorm, a=self.a, b=self.b, rng=self.rng)

            ctx = self._context(idx, gnorm, n_feat)
            # Warm-up: ép round-robin để bandit thu thập dữ liệu mọi arm trước khi khai thác
            if step < self.warmup:
                feat_id = step % n_arms
            else:
                feat_id = self.selector.choose(step, ctx)
            name = self.fe_names[feat_id]

            Z = self.Z["train"][name]
            dtrain = xgb.DMatrix(
                Z[idx], label=self._ytr[idx], weight=w,
                base_margin=Fm[idx].ravel(),
            )
            booster = xgb.train(base_params, dtrain, num_boost_round=1)

            contrib = booster.predict(self._dmat["train"][name],
                                      output_margin=True).reshape(-1, self.K)
            Fm += contrib
            self.models.append((booster, feat_id))
            self.feat_history.append(feat_id)

            if track_reward:
                Fm_val += booster.predict(self._dmat["val"][name],
                                          output_margin=True).reshape(-1, self.K)
                new_val_acc = (np.argmax(Fm_val, axis=1) == yval_i).mean()
                # Reward = Δaccuracy − λ·chi_phí_FE  (time-penalty)
                reward = (new_val_acc - prev_val_acc) - self.lam * self.fe_cost[feat_id]
                self.selector.reward(feat_id, reward, ctx)
                prev_val_acc = new_val_acc

        self.total_fe_cost = float(sum(self.fe_cost[f] for f in self.feat_history))
        return self

    def selection_frequency(self) -> dict[str, float]:
        """Tần suất chọn mỗi phép FE (toàn bộ boosting) — dữ liệu cho Hình 2."""
        c = np.bincount(self.feat_history, minlength=len(self.fe_names))
        return {n: float(v) / len(self.feat_history)
                for n, v in zip(self.fe_names, c)}

    # ---- dự đoán ---------------------------------------------------------- #
    def predict_margin(self, which: str = "test") -> np.ndarray:
        n = next(iter(self.Z[which].values())).shape[0]
        Fm = np.zeros((n, self.K))
        for booster, feat_id in self.models:
            name = self.fe_names[feat_id]
            Fm += booster.predict(self._dmat[which][name],
                                  output_margin=True).reshape(-1, self.K)
        return Fm

    def predict_proba(self, which: str = "test") -> np.ndarray:
        return _softmax(self.predict_margin(which))

    def predict(self, which: str = "test") -> np.ndarray:
        return np.argmax(self.predict_margin(which), axis=1)
