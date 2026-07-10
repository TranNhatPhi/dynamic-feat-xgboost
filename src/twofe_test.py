"""
ĐINH CUỐI — rotation CHỈ 2 phép rẻ nhất (ht_svd + random_proj) vs Bandit.

Nếu twofe ≈ bandit về accuracy mà cost ≤ bandit => bằng chứng đóng hoàn toàn:
"chỉ cần hard-code xoay vòng 2 phép rẻ, bộ máy bandit học online là vô dụng."
Dùng cho phần Discussion của bài Empirical Study (Hướng B).
"""

import csv
from pathlib import Path

import numpy as np

from src.fe.registry import FE_ARMS
from src.models.dynamic_feat import BanditSelector
from src.models.feat_xgboost import FeatXGBoost, FESelector
from src.preprocessing import prepare
from src.data_loader import UCIDataset
from src.select_datasets import hard_set

CFG = {"eta": 0.3, "max_depth": 6}
FOLDS = [0, 1, 2, 3]
SEEDS = [0, 1, 2]
ALPHA, LAM = 0.25, 0.1
N_BOOST = 100
OUT = Path(__file__).resolve().parent.parent / "results"
TWO = [FE_ARMS.index("ht_svd"), FE_ARMS.index("random_proj")]


class TwoFERotation(FESelector):
    def __init__(self, ids):
        self.ids = ids
    def choose(self, step, context):
        return self.ids[step % len(self.ids)]


def run(kind, sp, seed):
    sel = TwoFERotation(TWO) if kind == "twofe" else BanditSelector(FE_ARMS, alpha=ALPHA, seed=seed)
    m = FeatXGBoost(selector=sel, n_boost=N_BOOST, xgb_cfg=CFG, lam=LAM, warmup=10, seed=seed)
    m.fit(sp.X_train, sp.y_train, sp.X_val, sp.y_val, sp.X_test)
    return (m.predict("test") == sp.y_test).mean(), m.total_fe_cost


def paired(a, b):
    d = np.array(a) - np.array(b)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else float("inf")
    return d.mean(), d.std(ddof=1), t, (d > 1e-4).sum(), (np.abs(d) <= 1e-4).sum(), (d < -1e-4).sum()


def main():
    OUT.mkdir(exist_ok=True)
    per = {"twofe": {}, "bandit": {}}
    cost = {"twofe": {}, "bandit": {}}
    rows = [["dataset", "config", "fold", "seed", "acc", "cost"]]
    for name in hard_set():
        ds = UCIDataset(name)
        acc = {"twofe": [], "bandit": []}; cst = {"twofe": [], "bandit": []}
        for f in FOLDS:
            sp = prepare(ds.get_split(f))
            for s in SEEDS:
                for c in ("twofe", "bandit"):
                    a, cc = run(c, sp, s)
                    acc[c].append(a); cst[c].append(cc)
                    rows.append([name, c, f, s, round(a, 4), round(cc, 1)])
        for c in ("twofe", "bandit"):
            per[c][name] = np.mean(acc[c]); cost[c][name] = np.mean(cst[c])
        print(f"{name:28s} twofe={per['twofe'][name]:.3f} bandit={per['bandit'][name]:.3f} "
              f"| cost twofe={cost['twofe'][name]:.0f} BD={cost['bandit'][name]:.0f}", flush=True)

    with open(OUT / "twofe_long.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)
    names = list(per["twofe"].keys())
    m, sd, t, w, tie, l = paired([per["bandit"][n] for n in names], [per["twofe"][n] for n in names])
    ct, cb = np.mean([cost["twofe"][n] for n in names]), np.mean([cost["bandit"][n] for n in names])
    print(f"\n===== PAIRED (n={len(names)}) =====")
    print(f"  bandit vs twofe: Δacc={m:+.4f}±{sd:.4f} t={t:+.2f} (thắng/hòa/thua={w}/{tie}/{l})")
    print(f"  cost trung bình: twofe={ct:.1f}  bandit={cb:.1f}")
    print("\n>> Nếu Δ≈0 và twofe rẻ hơn/ngang => ĐINH CUỐI: bandit hoàn toàn vô dụng.")


if __name__ == "__main__":
    main()
