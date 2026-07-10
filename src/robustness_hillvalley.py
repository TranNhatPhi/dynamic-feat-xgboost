"""
Bước 1 (kiểm độ vững) — hill-valley qua 4 fold × 3 seed.

So Round-robin (gốc) vs Bandit (α=0.25, λ=0.1). Báo cáo mean ± std cho accuracy & cost,
và chênh lệch GHÉP CẶP (paired) bandit−RR trên từng (fold,seed) — kiểm xem chiến thắng
có ổn định hay chỉ do trúng seed đẹp.
"""

import numpy as np

from src.fe.registry import FE_ARMS
from src.models.dynamic_feat import BanditSelector
from src.models.feat_xgboost import FeatXGBoost, RoundRobinSelector
from src.preprocessing import UCIDataset, prepare

CFG = {"eta": 0.3, "max_depth": 6}
DATASET = "hill-valley"
FOLDS = [0, 1, 2, 3]
SEEDS = [0, 1, 2]
ALPHA, LAM = 0.25, 0.1


def run_one(split, selector, seed):
    m = FeatXGBoost(selector=selector, n_boost=100, xgb_cfg=CFG,
                    lam=LAM, warmup=10, seed=seed)
    m.fit(split.X_train, split.y_train, split.X_val, split.y_val, split.X_test)
    acc = (m.predict("test") == split.y_test).mean()
    return acc, m.total_fe_cost


def main():
    ds = UCIDataset(DATASET)
    rr_acc, rr_cost, bd_acc, bd_cost, diff = [], [], [], [], []

    print(f"Kiểm độ vững '{DATASET}' — {len(FOLDS)} fold × {len(SEEDS)} seed "
          f"(bandit α={ALPHA}, λ={LAM})\n")
    print(f"{'fold':>4} {'seed':>4} | {'RR acc':>7} {'BD acc':>7} {'Δacc':>7} "
          f"| {'RR cost':>7} {'BD cost':>7}")
    for f in FOLDS:
        split = prepare(ds.get_split(f))
        for s in SEEDS:
            a_rr, c_rr = run_one(split, RoundRobinSelector(len(FE_ARMS)), s)
            a_bd, c_bd = run_one(split, BanditSelector(FE_ARMS, alpha=ALPHA, seed=s), s)
            rr_acc.append(a_rr); rr_cost.append(c_rr)
            bd_acc.append(a_bd); bd_cost.append(c_bd)
            diff.append(a_bd - a_rr)
            print(f"{f:>4} {s:>4} | {a_rr:>7.4f} {a_bd:>7.4f} {a_bd-a_rr:>+7.4f} "
                  f"| {c_rr:>7.1f} {c_bd:>7.1f}")

    rr_acc, bd_acc, diff = map(np.array, (rr_acc, bd_acc, diff))
    rr_cost, bd_cost = np.array(rr_cost), np.array(bd_cost)
    print("\n===== TỔNG KẾT (mean ± std, n=%d) =====" % len(diff))
    print(f"  Round-robin  acc = {rr_acc.mean():.4f} ± {rr_acc.std():.4f}   "
          f"cost = {rr_cost.mean():.1f} ± {rr_cost.std():.1f}")
    print(f"  Bandit       acc = {bd_acc.mean():.4f} ± {bd_acc.std():.4f}   "
          f"cost = {bd_cost.mean():.1f} ± {bd_cost.std():.1f}")
    print(f"  Δacc (paired bandit−RR) = {diff.mean():+.4f} ± {diff.std():.4f}")
    cost_cut = (1 - bd_cost.mean() / rr_cost.mean()) * 100
    print(f"  Giảm cost trung bình = {cost_cut:.1f}%")

    # đọc nhanh ý nghĩa: mean thắng có lớn hơn std không?
    wins = (diff > 0).sum()
    print(f"\n  Bandit thắng RR ở {wins}/{len(diff)} lần chạy.")
    if diff.mean() > 0 and diff.mean() > diff.std():
        print("  => Chênh lệch ỔN ĐỊNH (mean > std): chiến thắng đáng tin.")
    elif diff.mean() > 0:
        print("  => Bandit nhỉnh hơn nhưng std lớn hơn mean: CHƯA chắc chắn về accuracy "
              "(nhưng cost giảm mạnh là chắc).")
    else:
        print("  => Bandit KHÔNG thắng accuracy ổn định. Xem lại (cost vẫn là điểm mạnh).")


if __name__ == "__main__":
    main()
