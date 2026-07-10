"""
Bước 2 — Quét HARD_SET để tìm "sân nhà" + gom dữ liệu Hình 2 (heatmap chọn FE).

Với mỗi dataset (4 fold, 1 seed): chạy plain-XGBoost, Round-robin (gốc), Bandit (α=0.25,λ=0.1).
Ghi:
  results/hardset_scan.csv        : acc (mean±std) & cost của 3 cấu hình / dataset
  results/hardset_bandit_freq.csv : tần suất bandit chọn mỗi FE / dataset (cho Hình 2)
"""

import csv
import time
from pathlib import Path

import numpy as np
from xgboost import XGBClassifier

from src.fe.registry import FE_ARMS
from src.models.dynamic_feat import BanditSelector
from src.models.feat_xgboost import FeatXGBoost, RoundRobinSelector
from src.preprocessing import prepare
from src.data_loader import UCIDataset
from src.select_datasets import hard_set

CFG = {"eta": 0.3, "max_depth": 6}
FOLDS = [0, 1, 2, 3]
SEED = 0
ALPHA, LAM = 0.25, 0.1
OUT = Path(__file__).resolve().parent.parent / "results"


def plain_xgb(split):
    m = XGBClassifier(n_estimators=100, max_depth=6, eta=0.3, verbosity=0)
    m.fit(split.X_train, split.y_train)
    return (m.predict(split.X_test) == split.y_test).mean()


def feat_run(split, selector, seed):
    m = FeatXGBoost(selector=selector, n_boost=100, xgb_cfg=CFG,
                    lam=LAM, warmup=10, seed=seed)
    m.fit(split.X_train, split.y_train, split.X_val, split.y_val, split.X_test)
    acc = (m.predict("test") == split.y_test).mean()
    return acc, m.total_fe_cost, m.selection_frequency()


def main():
    datasets = hard_set()
    scan_rows, freq_rows = [], []
    print(f"Quét {len(datasets)} tập HARD_SET ({len(FOLDS)} fold, seed={SEED})\n")
    print(f"{'dataset':28s} {'plainXGB':>9} {'round-r':>9} {'bandit':>9} "
          f"{'RRcost':>7} {'BDcost':>7} {'sanNha?':>8}")

    for name in datasets:
        try:
            ds = UCIDataset(name)
            px, rr, bd, rrc, bdc = [], [], [], [], []
            freq_acc = np.zeros(len(FE_ARMS))
            t0 = time.time()
            for f in FOLDS:
                sp = prepare(ds.get_split(f))
                px.append(plain_xgb(sp))
                a, c, _ = feat_run(sp, RoundRobinSelector(len(FE_ARMS)), SEED)
                rr.append(a); rrc.append(c)
                a, c, fr = feat_run(sp, BanditSelector(FE_ARMS, alpha=ALPHA, seed=SEED), SEED)
                bd.append(a); bdc.append(c)
                freq_acc += np.array([fr[n] for n in FE_ARMS])
            px, rr, bd = map(np.array, (px, rr, bd))
            freq = freq_acc / len(FOLDS)
            dt = time.time() - t0

            # "sân nhà": FE-boosting (bandit hoặc RR) >= plain XGB  &  bandit >= RR
            home = (max(bd.mean(), rr.mean()) >= px.mean() - 1e-9) and (bd.mean() >= rr.mean() - 1e-9)
            tag = "YES" if home else "-"
            print(f"{name:28s} {px.mean():9.4f} {rr.mean():9.4f} {bd.mean():9.4f} "
                  f"{np.mean(rrc):7.1f} {np.mean(bdc):7.1f} {tag:>8}  ({dt:.0f}s)")

            scan_rows.append({
                "dataset": name,
                "plain_xgb_acc": round(px.mean(), 4),
                "round_robin_acc": round(rr.mean(), 4), "round_robin_acc_std": round(rr.std(), 4),
                "bandit_acc": round(bd.mean(), 4), "bandit_acc_std": round(bd.std(), 4),
                "round_robin_cost": round(np.mean(rrc), 1),
                "bandit_cost": round(np.mean(bdc), 1),
                "cost_cut_pct": round((1 - np.mean(bdc) / np.mean(rrc)) * 100, 1),
                "home_ground": home,
            })
            for n, fv in zip(FE_ARMS, freq):
                freq_rows.append({"dataset": name, "fe": n, "freq": round(float(fv), 4)})
        except Exception as e:
            print(f"{name:28s} LỖI: {repr(e)[:60]}")

    OUT.mkdir(exist_ok=True)
    with open(OUT / "hardset_scan.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(scan_rows[0].keys()))
        w.writeheader(); w.writerows(scan_rows)
    with open(OUT / "hardset_bandit_freq.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "fe", "freq"])
        w.writeheader(); w.writerows(freq_rows)

    n_home = sum(r["home_ground"] for r in scan_rows)
    avg_cut = np.mean([r["cost_cut_pct"] for r in scan_rows])
    print(f"\n===== TỔNG KẾT =====")
    print(f"  'Sân nhà' (bandit>=RR & FE-boost>=plainXGB): {n_home}/{len(scan_rows)} tập")
    print(f"  Giảm cost trung bình (bandit vs RR): {avg_cut:.1f}%")
    print(f"  Đã ghi: results/hardset_scan.csv, results/hardset_bandit_freq.csv")


if __name__ == "__main__":
    main()
