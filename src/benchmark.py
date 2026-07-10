"""
Bảng 2 — Benchmark RUNTIME & MEMORY THẬT (thay cost-proxy).

Đo công bằng: mỗi cấu hình CHỈ precompute các FE nó thực sự dùng (đúng chi phí thật).
QUAN TRỌNG: bandit phải precompute CẢ 6 FE (thực đơn để chọn) — kể cả autofeat đắt —
nên chi phí precompute thật của nó gồm cả autofeat, dù hiếm khi chọn.

Đo trên fold 0, 3 lần lặp (mean±std):
  - fit_time_s : wall-clock của .fit() (gồm precompute FE + boosting)
  - fe_mem_mb  : tổng dung lượng các ma trận FE đã precompute (train+val+test)
Cùng số luồng cho mọi config (công bằng tương đối). Ghi results/benchmark.csv.
"""

import csv
import time
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
N_BOOST = 100
REPEATS = 3
SEED = 0
OUT = Path(__file__).resolve().parent.parent / "results"

CHEAP5 = ["identity", "random_proj", "ht_svd", "robust", "minmax"]
TWO = ["ht_svd", "random_proj"]

# mỗi config: (fe_names dùng thật, hàm tạo selector theo số arm)
CONFIGS = {
    "plain_xgb":   None,
    "fixed_htsvd": (["ht_svd"], lambda n: FixedSelector(0)),
    "twofe":       (TWO, lambda n: RoundRobinSelector(n)),
    "cheap_rr":    (CHEAP5, lambda n: RoundRobinSelector(n)),
    "round_robin": (FE_ARMS, lambda n: RoundRobinSelector(n)),
    # bandit: PHẢI có cả 6 FE làm thực đơn
    "bandit":      (FE_ARMS, lambda n: BanditSelector(FE_ARMS, alpha=0.25, seed=SEED)),
}


def fe_mem_mb(m):
    total = 0
    for which in ("train", "val", "test"):
        for Z in m.Z[which].values():
            total += Z.nbytes
    return total / 1e6


def bench_one(name, sp):
    times = []
    mem = 0.0
    for _ in range(REPEATS):
        if name == "plain_xgb":
            model = XGBClassifier(n_estimators=N_BOOST, max_depth=6, eta=0.3, verbosity=0)
            t = time.perf_counter()
            model.fit(sp.X_train, sp.y_train)
            times.append(time.perf_counter() - t)
            mem = sp.X_train.nbytes * 3 / 1e6  # xấp xỉ: chỉ giữ data gốc
        else:
            fe_names, mk = CONFIGS[name]
            m = FeatXGBoost(selector=mk(len(fe_names)), n_boost=N_BOOST, xgb_cfg=CFG,
                            lam=0.1, warmup=min(10, N_BOOST), fe_names=fe_names, seed=SEED)
            t = time.perf_counter()
            m.fit(sp.X_train, sp.y_train, sp.X_val, sp.y_val, sp.X_test)
            times.append(time.perf_counter() - t)
            mem = fe_mem_mb(m)
    return np.mean(times), np.std(times), mem


def main():
    OUT.mkdir(exist_ok=True)
    rows = [["dataset", "config", "fit_time_s", "fit_time_std", "fe_mem_mb"]]
    print(f"{'dataset':26s} {'config':12s} {'time(s)':>9} {'mem(MB)':>9}")
    for name in hard_set():
        sp = prepare(UCIDataset(name).get_split(0))
        for cfg in CONFIGS:
            mt, st, mem = bench_one(cfg, sp)
            rows.append([name, cfg, round(mt, 3), round(st, 3), round(mem, 2)])
            print(f"{name:26s} {cfg:12s} {mt:9.2f} {mem:9.2f}", flush=True)
    with open(OUT / "benchmark.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows)

    # tóm tắt trung bình / config
    import collections
    agg_t = collections.defaultdict(list); agg_m = collections.defaultdict(list)
    for r in rows[1:]:
        agg_t[r[1]].append(r[2]); agg_m[r[1]].append(r[4])
    print("\n=== TRUNG BÌNH toàn bộ ===")
    print(f"{'config':12s} {'time(s)':>9} {'mem(MB)':>9}")
    for c in CONFIGS:
        print(f"{c:12s} {np.mean(agg_t[c]):9.2f} {np.mean(agg_m[c]):9.2f}")
    print("\nĐã ghi results/benchmark.csv")


if __name__ == "__main__":
    main()
