"""
PHÉP THỬ SỐNG-CHẾT: Bandit có hơn Fixed-ht_svd (luôn dùng ht_svd) không?

Vì bandit gần như luôn bốc ht_svd, phải kiểm: nếu Fixed-ht_svd ≈ bandit về accuracy
mà lại RẺ HƠN (không warm-up, không explore) => contextual bandit là thừa => mất novelty.
"""

import numpy as np

from src.fe.registry import FE_ARMS
from src.models.dynamic_feat import BanditSelector
from src.models.feat_xgboost import FeatXGBoost, FixedSelector, RoundRobinSelector
from src.preprocessing import prepare
from src.data_loader import UCIDataset

CFG = {"eta": 0.3, "max_depth": 6}
FOLDS = [0, 1, 2, 3]
SEED = 0
ALPHA, LAM = 0.25, 0.1
DATASETS = ["hill-valley", "monks-3", "libras", "arrhythmia", "semeion",
            "conn-bench-sonar-mines-rocks"]


def run(split, selector):
    m = FeatXGBoost(selector=selector, n_boost=100, xgb_cfg=CFG,
                    lam=LAM, warmup=10, seed=SEED)
    m.fit(split.X_train, split.y_train, split.X_val, split.y_val, split.X_test)
    return (m.predict("test") == split.y_test).mean(), m.total_fe_cost


def main():
    ht = FE_ARMS.index("ht_svd")
    print(f"So Fixed-ht_svd vs Bandit (α={ALPHA}, λ={LAM}), {len(FOLDS)} fold\n")
    print(f"{'dataset':28s} {'RR':>7} {'FixHT':>7} {'Bandit':>7} | "
          f"{'ΔBD-FixHT':>10} | {'FixHTcost':>9} {'BDcost':>7}")
    diffs = []
    for name in DATASETS:
        ds = UCIDataset(name)
        rr, fx, bd, fxc, bdc = [], [], [], [], []
        for f in FOLDS:
            sp = prepare(ds.get_split(f))
            a, _ = run(sp, RoundRobinSelector(len(FE_ARMS))); rr.append(a)
            a, c = run(sp, FixedSelector(ht)); fx.append(a); fxc.append(c)
            a, c = run(sp, BanditSelector(FE_ARMS, alpha=ALPHA, seed=SEED)); bd.append(a); bdc.append(c)
        rr, fx, bd = np.mean(rr), np.mean(fx), np.mean(bd)
        d = bd - fx
        diffs.append(d)
        print(f"{name:28s} {rr:7.4f} {fx:7.4f} {bd:7.4f} | {d:>+10.4f} | "
              f"{np.mean(fxc):9.1f} {np.mean(bdc):7.1f}")

    diffs = np.array(diffs)
    print(f"\nΔ(Bandit − Fixed-ht_svd) trung bình = {diffs.mean():+.4f} ± {diffs.std():.4f}")
    print(f"Bandit hơn Fixed-ht_svd ở {(diffs>0).sum()}/{len(diffs)} tập.")
    if diffs.mean() <= 0.002:
        print(">> KẾT LUẬN: Bandit KHÔNG hơn Fixed-ht_svd đáng kể => contextual bandit bị đe doạ novelty.")
    else:
        print(">> KẾT LUẬN: Bandit có nhỉnh hơn Fixed-ht_svd => còn lý do tồn tại.")


if __name__ == "__main__":
    main()
