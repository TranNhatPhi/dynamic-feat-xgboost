"""
HƯỚNG B — So sánh hệ thống cho bài nghiên cứu thực nghiệm.

4 cấu hình × 15 tập HARD_SET × 4 fold × 3 seed:
  plain_xgb    : XGBoost thường (mốc ngoài FE-boosting)
  round_robin  : Feat-XGBoost gốc (xoay vòng i%6)
  fixed_htsvd  : luôn dùng HT-SVD (baseline đơn giản đã "hạ" bandit)
  bandit       : contextual bandit (giữ như baseline BỊ BÁC BỎ trong bài)

Ghi results/study_long.csv (dataset,config,fold,seed,acc,cost) — append theo từng dataset
để không mất dữ liệu nếu bị ngắt. In tóm tắt + kiểm định ghép cặp ở cuối.
"""

import csv
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from src.fe.registry import FE_ARMS
from src.models.dynamic_feat import BanditSelector
from src.models.feat_xgboost import FeatXGBoost, FixedSelector, RoundRobinSelector
from src.preprocessing import prepare
from src.data_loader import UCIDataset
from src.select_datasets import hard_set

CFG = {"eta": 0.3, "max_depth": 6}
FOLDS = [0, 1, 2, 3]
SEEDS = [0, 1, 2]
ALPHA, LAM = 0.25, 0.1
N_BOOST = 100
OUT = Path(__file__).resolve().parent.parent / "results"
LONG = OUT / "study_long.csv"
HT = FE_ARMS.index("ht_svd")


def make_config(name, seed):
    if name == "round_robin":
        return RoundRobinSelector(len(FE_ARMS))
    if name == "fixed_htsvd":
        return FixedSelector(HT)
    if name == "bandit":
        return BanditSelector(FE_ARMS, alpha=ALPHA, seed=seed)
    return None


def eval_run(name, sp, seed):
    if name == "plain_xgb":
        m = XGBClassifier(n_estimators=N_BOOST, max_depth=6, eta=0.3,
                          verbosity=0, random_state=seed)
        m.fit(sp.X_train, sp.y_train)
        return (m.predict(sp.X_test) == sp.y_test).mean(), 0.0
    m = FeatXGBoost(selector=make_config(name, seed), n_boost=N_BOOST,
                    xgb_cfg=CFG, lam=LAM, warmup=10, seed=seed)
    m.fit(sp.X_train, sp.y_train, sp.X_val, sp.y_val, sp.X_test)
    return (m.predict("test") == sp.y_test).mean(), m.total_fe_cost


def paired(a, b):
    """mean diff (a−b), std, t-stat, thắng/hòa/thua theo từng dataset (a vs b)."""
    d = np.array(a) - np.array(b)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else float("inf")
    return d.mean(), d.std(ddof=1), t, (d > 1e-4).sum(), (np.abs(d) <= 1e-4).sum(), (d < -1e-4).sum()


def main():
    OUT.mkdir(exist_ok=True)
    configs = ["plain_xgb", "round_robin", "fixed_htsvd", "bandit"]
    with open(LONG, "w", newline="") as f:
        csv.writer(f).writerow(["dataset", "config", "fold", "seed", "acc", "cost"])

    per_ds = {c: {} for c in configs}  # config -> {dataset -> mean acc}
    for name in hard_set():
        ds = UCIDataset(name)
        rows, accs = [], {c: [] for c in configs}
        costs = {c: [] for c in configs}
        for f in FOLDS:
            sp = prepare(ds.get_split(f))
            for s in SEEDS:
                for c in configs:
                    acc, cost = eval_run(c, sp, s)
                    accs[c].append(acc); costs[c].append(cost)
                    rows.append([name, c, f, s, round(acc, 4), round(cost, 1)])
        with open(LONG, "a", newline="") as fh:
            csv.writer(fh).writerows(rows)
        line = f"{name:28s}"
        for c in configs:
            per_ds[c][name] = np.mean(accs[c])
            line += f" {c[:5]}={np.mean(accs[c]):.3f}"
        line += f" | cost RR={np.mean(costs['round_robin']):.0f} " \
                f"FixHT={np.mean(costs['fixed_htsvd']):.0f} BD={np.mean(costs['bandit']):.0f}"
        print(line, flush=True)

    # ---- kiểm định ghép cặp (theo dataset, dùng mean acc mỗi dataset) ----
    names = list(per_ds["plain_xgb"].keys())
    def vec(c): return [per_ds[c][n] for n in names]
    print("\n===== KIỂM ĐỊNH GHÉP CẶP (n=%d dataset) =====" % len(names))
    for a, b in [("fixed_htsvd", "round_robin"), ("bandit", "fixed_htsvd"),
                 ("bandit", "round_robin"), ("round_robin", "plain_xgb")]:
        m, sd, t, w, tie, l = paired(vec(a), vec(b))
        print(f"  {a:12s} vs {b:12s}: Δacc={m:+.4f}±{sd:.4f} t={t:+.2f} "
              f"(thắng/hòa/thua = {w}/{tie}/{l})")
    print("\nĐã ghi results/study_long.csv")


if __name__ == "__main__":
    main()
