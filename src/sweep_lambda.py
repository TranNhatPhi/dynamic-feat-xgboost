"""
Bước 13b — Sweep λ (time-penalty) × α (exploration) cho bandit, so với các mốc.

Mục đích:
  1. Đường cong đánh đổi accuracy vs cost khi tăng λ (Hình 3 của paper).
  2. Giảm α để bandit khai thác mạnh hơn -> có bốc FE rẻ không?
  3. Mốc Fixed-identity (sanity check): identity thuần có ≈ XGBoost thường không?
"""

import numpy as np
from xgboost import XGBClassifier

from src.fe.registry import FE_ARMS
from src.models.dynamic_feat import BanditSelector
from src.models.feat_xgboost import FeatXGBoost, FixedSelector, RoundRobinSelector
from src.preprocessing import load_prepared

CFG = {"eta": 0.3, "max_depth": 6}
LAMBDAS = [0.01, 0.05, 0.1, 0.3]
ALPHAS = [1.0, 0.25]


def acc_cost(sel, s, lam=0.01):
    m = FeatXGBoost(selector=sel, n_boost=100, xgb_cfg=CFG, lam=lam, warmup=10, seed=1)
    m.fit(s.X_train, s.y_train, s.X_val, s.y_val, s.X_test)
    acc = (m.predict("test") == s.y_test).mean()
    freq = m.selection_frequency()
    top = ", ".join(f"{k}={v*100:.0f}%" for k, v in
                    sorted(freq.items(), key=lambda kv: -kv[1])[:2])
    return acc, m.total_fe_cost, top


def run(name):
    s, _ = load_prepared(name, 0)
    print(f"\n===== {name} (train={s.X_train.shape}, K={int(s.y_train.max()+1)}) =====")

    xg = XGBClassifier(n_estimators=100, max_depth=6, eta=0.3, verbosity=0)
    xg.fit(s.X_train, s.y_train)
    print(f"  [ref] XGBoost thuong    acc={(xg.predict(s.X_test)==s.y_test).mean():.4f}")

    a, c, _ = acc_cost(RoundRobinSelector(len(FE_ARMS)), s)
    print(f"  [ref] Round-robin       acc={a:.4f}  cost={c:6.1f}")
    a, c, _ = acc_cost(FixedSelector(FE_ARMS.index("identity")), s)
    print(f"  [ref] Fixed-identity    acc={a:.4f}  cost={c:6.1f}  (sanity vs XGBoost)")
    a, c, _ = acc_cost(FixedSelector(FE_ARMS.index("random_proj")), s)
    print(f"  [ref] Fixed-cheap(rp)   acc={a:.4f}  cost={c:6.1f}")

    print(f"  {'--- Bandit sweep ---':22s} {'acc':>6s} {'cost':>7s}  top-FE")
    for alpha in ALPHAS:
        for lam in LAMBDAS:
            sel = BanditSelector(FE_ARMS, alpha=alpha, seed=1)
            a, c, top = acc_cost(sel, s, lam=lam)
            print(f"  a={alpha:<4} lam={lam:<5}        {a:6.4f} {c:7.1f}  {top}")


if __name__ == "__main__":
    for name in ["hill-valley", "arrhythmia"]:
        run(name)
