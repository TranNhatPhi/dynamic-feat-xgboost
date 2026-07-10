"""
PHÉP THỬ CHỐT ĐỊNH VỊ: bandit có hơn 'cheap-round-robin' (xoay vòng CHỈ các FE rẻ) không?

cheap-round-robin = round-robin trên [identity, random_proj, ht_svd, robust, minmax]
(bỏ autofeat — phép đắt nhất). Nếu nó ≈ bandit ở cùng chi phí => bandit thừa (chỉ cần bỏ
autofeat thủ công). Nếu bandit > nó => bandit justified.

So 3 cấu hình: cheap_rr / bandit / (round_robin để tham chiếu), 15 tập × 4 fold × 3 seed.
Ghi results/cheap_rr_long.csv. In paired t-test.
"""

import csv
from pathlib import Path

import numpy as np

from src.fe.registry import FE_ARMS
from src.models.dynamic_feat import BanditSelector
from src.models.feat_xgboost import FeatXGBoost, FESelector, RoundRobinSelector
from src.preprocessing import prepare
from src.data_loader import UCIDataset
from src.select_datasets import hard_set

CFG = {"eta": 0.3, "max_depth": 6}
FOLDS = [0, 1, 2, 3]
SEEDS = [0, 1, 2]
ALPHA, LAM = 0.25, 0.1
N_BOOST = 100
OUT = Path(__file__).resolve().parent.parent / "results"

CHEAP = [FE_ARMS.index(n) for n in ["identity", "random_proj", "ht_svd", "robust", "minmax"]]


class CheapRoundRobin(FESelector):
    """Xoay vòng chỉ trên các FE rẻ (bỏ autofeat)."""
    def __init__(self, cheap_ids):
        self.ids = cheap_ids
    def choose(self, step, context):
        return self.ids[step % len(self.ids)]


def run(name, sp, seed):
    if name == "cheap_rr":
        sel = CheapRoundRobin(CHEAP)
    elif name == "bandit":
        sel = BanditSelector(FE_ARMS, alpha=ALPHA, seed=seed)
    else:
        sel = RoundRobinSelector(len(FE_ARMS))
    m = FeatXGBoost(selector=sel, n_boost=N_BOOST, xgb_cfg=CFG, lam=LAM, warmup=10, seed=seed)
    m.fit(sp.X_train, sp.y_train, sp.X_val, sp.y_val, sp.X_test)
    return (m.predict("test") == sp.y_test).mean(), m.total_fe_cost


def paired(a, b):
    d = np.array(a) - np.array(b)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else float("inf")
    return d.mean(), d.std(ddof=1), t, (d > 1e-4).sum(), (np.abs(d) <= 1e-4).sum(), (d < -1e-4).sum()


def main():
    OUT.mkdir(exist_ok=True)
    configs = ["cheap_rr", "bandit", "round_robin"]
    per = {c: {} for c in configs}
    costs = {c: {} for c in configs}
    rows = [["dataset", "config", "fold", "seed", "acc", "cost"]]
    for name in hard_set():
        ds = UCIDataset(name)
        acc = {c: [] for c in configs}; cst = {c: [] for c in configs}
        for f in FOLDS:
            sp = prepare(ds.get_split(f))
            for s in SEEDS:
                for c in configs:
                    a, cc = run(c, sp, s)
                    acc[c].append(a); cst[c].append(cc)
                    rows.append([name, c, f, s, round(a, 4), round(cc, 1)])
        for c in configs:
            per[c][name] = np.mean(acc[c]); costs[c][name] = np.mean(cst[c])
        print(f"{name:28s} cheap={per['cheap_rr'][name]:.3f} bandit={per['bandit'][name]:.3f} "
              f"RR={per['round_robin'][name]:.3f} | cost cheap={costs['cheap_rr'][name]:.0f} "
              f"BD={costs['bandit'][name]:.0f}", flush=True)

    with open(OUT / "cheap_rr_long.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    names = list(per["cheap_rr"].keys())
    def vec(c, d): return [d[c][n] for n in names]
    print("\n===== PAIRED (n=%d) =====" % len(names))
    m, sd, t, w, tie, l = paired(vec("bandit", per), vec("cheap_rr", per))
    print(f"  bandit vs cheap_rr : Δacc={m:+.4f}±{sd:.4f} t={t:+.2f} (thắng/hòa/thua={w}/{tie}/{l})")
    ba = np.mean([costs["bandit"][n] for n in names])
    ch = np.mean([costs["cheap_rr"][n] for n in names])
    print(f"  cost trung bình: cheap_rr={ch:.1f}  bandit={ba:.1f}")
    print("\n>> Nếu Δ≈0 và cheap_rr rẻ hơn/ngang => bandit THỪA. Nếu bandit thắng rõ => justified.")


if __name__ == "__main__":
    main()
