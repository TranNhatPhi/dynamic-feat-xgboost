"""
Agent bandit chọn động phép feature engineering trong từng bước boosting.

Cung cấp 3 chiến lược (để làm ablation):
  - EpsilonGreedy : MAB thuần, context-free (baseline đơn giản)
  - UCB1          : MAB thuần, context-free
  - LinUCB        : CONTEXTUAL bandit — chọn FE theo đặc trưng tập mẫu khó  [TRỤ CHÍNH]

Mỗi "arm" là một phép FE (Identity, RandomProjection, HT-SVD, RobustScaler,
MinMaxScaler, Autofeat). Ở mỗi bước boosting:
    context x  = vector mô tả tập mẫu khó hiện tại (chuẩn hoá)
    arm        = agent.select(x)          -> chọn 1 phép FE
    (áp FE, dựng cây, đo cải thiện)
    agent.update(arm, reward, x)          -> học

LƯU Ý HỌC THUẬT: đây là *contextual bandit*, KHÔNG phải RL đầy đủ
(không có chuyển trạng thái / credit assignment dài hạn). Gọi đúng tên trong paper.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BanditBase(ABC):
    """Giao diện chung cho mọi agent chọn FE."""

    def __init__(self, arms: list[str], seed: int = 0):
        self.arms = list(arms)
        self.n_arms = len(arms)
        self.rng = np.random.default_rng(seed)
        self.counts = np.zeros(self.n_arms, dtype=np.int64)
        self.history: list[int] = []  # log arm đã chọn -> phục vụ interpretability

    @abstractmethod
    def select(self, context: np.ndarray | None = None) -> int:
        ...

    @abstractmethod
    def update(self, arm: int, reward: float, context: np.ndarray | None = None) -> None:
        ...

    def selection_frequency(self) -> dict[str, float]:
        """Tần suất mỗi phép FE được chọn — dùng cho phần diễn giải chính sách."""
        total = max(len(self.history), 1)
        freq = np.bincount(self.history, minlength=self.n_arms) / total
        return {name: float(f) for name, f in zip(self.arms, freq)}


class EpsilonGreedy(BanditBase):
    """MAB thuần: với xác suất eps chọn ngẫu nhiên, còn lại chọn arm tốt nhất."""

    def __init__(self, arms, eps: float = 0.1, seed: int = 0):
        super().__init__(arms, seed)
        self.eps = eps
        self.values = np.zeros(self.n_arms)  # ước lượng reward trung bình

    def select(self, context=None) -> int:
        if self.rng.random() < self.eps:
            arm = int(self.rng.integers(self.n_arms))
        else:
            arm = int(np.argmax(self.values))
        self.history.append(arm)
        return arm

    def update(self, arm, reward, context=None):
        self.counts[arm] += 1
        # trung bình tăng dần
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class UCB1(BanditBase):
    """MAB thuần: chọn theo cận trên tin cậy (upper confidence bound)."""

    def __init__(self, arms, c: float = 2.0, seed: int = 0):
        super().__init__(arms, seed)
        self.c = c
        self.values = np.zeros(self.n_arms)

    def select(self, context=None) -> int:
        # thử mỗi arm ít nhất 1 lần
        for a in range(self.n_arms):
            if self.counts[a] == 0:
                self.history.append(a)
                return a
        t = self.counts.sum()
        bonus = np.sqrt(self.c * np.log(t) / self.counts)
        arm = int(np.argmax(self.values + bonus))
        self.history.append(arm)
        return arm

    def update(self, arm, reward, context=None):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]


class LinUCB(BanditBase):
    """
    Contextual bandit (LinUCB, disjoint model — Li et al. 2010).

    Mỗi arm học một vector trọng số theta_a; ước lượng reward = theta_a . x,
    cộng thêm phần thưởng khám phá alpha * sqrt(x^T A_a^{-1} x).
    """

    def __init__(self, arms, context_dim: int, alpha: float = 1.0, seed: int = 0):
        super().__init__(arms, seed)
        self.d = context_dim
        self.alpha = alpha
        # A_a = I (d x d), b_a = 0 (d,)
        self.A = [np.eye(self.d) for _ in range(self.n_arms)]
        self.b = [np.zeros(self.d) for _ in range(self.n_arms)]

    def select(self, context: np.ndarray) -> int:
        x = np.asarray(context, dtype=float).reshape(-1)
        assert x.shape[0] == self.d, f"context dim {x.shape[0]} != {self.d}"
        scores = np.empty(self.n_arms)
        for a in range(self.n_arms):
            A_inv = np.linalg.inv(self.A[a])
            theta = A_inv @ self.b[a]
            scores[a] = theta @ x + self.alpha * np.sqrt(x @ A_inv @ x)
        arm = int(np.argmax(scores))
        self.history.append(arm)
        return arm

    def update(self, arm: int, reward: float, context: np.ndarray):
        x = np.asarray(context, dtype=float).reshape(-1)
        self.A[arm] += np.outer(x, x)
        self.b[arm] += reward * x
        self.counts[arm] += 1


def make_bandit(name: str, arms: list[str], context_dim: int, seed: int = 0) -> BanditBase:
    """Factory để chọn chiến lược khi chạy ablation."""
    name = name.lower()
    if name in ("epsilon", "epsilon-greedy", "eps"):
        return EpsilonGreedy(arms, seed=seed)
    if name == "ucb1":
        return UCB1(arms, seed=seed)
    if name in ("linucb", "contextual"):
        return LinUCB(arms, context_dim=context_dim, seed=seed)
    raise ValueError(f"Chiến lược bandit không rõ: {name}")


# --------------------------------------------------------------------------- #
# Self-test: mô phỏng bài toán chọn FE mà arm tốt nhất PHỤ THUỘC context.
# Kỳ vọng: LinUCB (contextual) tiệm cận oracle; UCB1/eps (context-free) kém hơn
# vì không nhìn được context. Chứng minh cơ chế học đúng TRƯỚC khi cắm vào XGBoost.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    FE_ARMS = ["identity", "random_proj", "ht_svd", "robust", "minmax", "autofeat"]
    D = 4  # context: [n_samples, n_features, noise_level, imbalance] (đã chuẩn hoá)
    N_STEPS = 4000
    rng = np.random.default_rng(42)

    # "sự thật": mỗi arm tốt cho một vùng context khác nhau.
    true_w = rng.normal(size=(len(FE_ARMS), D))

    def sample_context():
        return rng.random(D)

    def reward_of(arm, ctx):
        # reward kỳ vọng = sigmoid(w_arm . ctx), có nhiễu Bernoulli
        p = 1.0 / (1.0 + np.exp(-true_w[arm] @ ctx))
        return float(rng.random() < p)

    results = {}
    for strat in ["epsilon", "ucb1", "linucb"]:
        agent = make_bandit(strat, FE_ARMS, context_dim=D, seed=1)
        total, oracle = 0.0, 0.0
        for _ in range(N_STEPS):
            ctx = sample_context()
            arm = agent.select(ctx)
            r = reward_of(arm, ctx)
            agent.update(arm, r, ctx)
            total += r
            # oracle = arm có reward kỳ vọng cao nhất cho context này
            best = max(range(len(FE_ARMS)),
                       key=lambda a: 1.0 / (1.0 + np.exp(-true_w[a] @ ctx)))
            oracle += 1.0 / (1.0 + np.exp(-true_w[best] @ ctx))
        results[strat] = (total, oracle, agent)

    print(f"Mô phỏng {N_STEPS} bước, {len(FE_ARMS)} arm, context {D} chiều\n")
    print(f"{'chiến lược':12s} {'reward TB':>10s} {'so với oracle':>14s}")
    for strat, (total, oracle, _) in results.items():
        print(f"{strat:12s} {total/N_STEPS:10.3f} {total/oracle*100:13.1f}%")

    print("\nTần suất chọn FE của LinUCB (interpretable policy):")
    for name, f in results["linucb"][2].selection_frequency().items():
        print(f"  {name:14s} {f*100:5.1f}%")
