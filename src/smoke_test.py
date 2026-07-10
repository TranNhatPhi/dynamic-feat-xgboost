"""Smoke test end-to-end: 4 cấu hình ablation vs XGBoost thường, trên vài dataset nhỏ."""

import time

import numpy as np
from xgboost import XGBClassifier

from src.fe.registry import FE_ARMS
from src.models.dynamic_feat import BanditSelector
from src.models.feat_xgboost import (
    FeatXGBoost,
    FixedSelector,
    RandomSelector,
    RoundRobinSelector,
)
from src.preprocessing import load_prepared

CFG = {"eta": 0.3, "max_depth": 6}


def run(name: str):
    s, _ = load_prepared(name, 0)
    K = int(s.y_train.max() + 1)
    print(f"===== {name} (train={s.X_train.shape}, K={K}) =====")

    x = XGBClassifier(n_estimators=100, max_depth=6, eta=0.3, verbosity=0)
    x.fit(s.X_train, s.y_train)
    acc = (x.predict(s.X_test) == s.y_test).mean()
    print(f"  {'XGBoost thuong':26s} acc={acc:.4f}")

    sels = {
        "C. Round-robin (GOC)": RoundRobinSelector(len(FE_ARMS)),
        "A. Fixed random_proj": FixedSelector(FE_ARMS.index("random_proj")),
        "B. Random FE": RandomSelector(len(FE_ARMS), seed=1),
        "D. Bandit (NOVELTY)": BanditSelector(FE_ARMS, alpha=1.0, seed=1),
    }
    for tag, sel in sels.items():
        t = time.time()
        m = FeatXGBoost(selector=sel, n_boost=100, xgb_cfg=CFG,
                        lam=0.01, warmup=10, seed=1)
        m.fit(s.X_train, s.y_train, s.X_val, s.y_val, s.X_test)
        acc = (m.predict("test") == s.y_test).mean()
        dt = time.time() - t
        print(f"  {tag:26s} acc={acc:.4f}  cost_FE={m.total_fe_cost:6.1f}  ({dt:.1f}s)")
        if isinstance(sel, BanditSelector):
            freq = m.selection_frequency()
            top = sorted(freq.items(), key=lambda kv: -kv[1])[:3]
            line = ", ".join(f"{k}={v*100:.0f}%" for k, v in top)
            print(f"     bandit chon nhieu nhat: {line}")
    print()


if __name__ == "__main__":
    for name in ["wine", "iris"]:
        run(name)
