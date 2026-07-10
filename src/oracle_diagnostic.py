"""
Hướng A — Phân tích ORACLE: trần trên của mọi bộ chọn FE động.

Mỗi bước boosting, thử CẢ 6 FE, chọn phép giảm val-loss nhiều nhất (greedy oracle).
  - Nếu FE tối ưu ĐỔI theo bước (early vs late) => có cấu trúc thời gian để bandit khai thác.
  - Nếu oracle KHÔNG vượt Fixed-ht_svd => ngay cả bộ chọn hoàn hảo cũng vô dụng => Hướng B.
"""

import numpy as np
import xgboost as xgb

from src.fe.registry import FE_ARMS
from src.models.feat_xgboost import (
    FeatXGBoost, FixedSelector, RoundRobinSelector, _logloss, _softmax,
)
from src.goss import goss_sample
from src.preprocessing import prepare
from src.data_loader import UCIDataset

CFG = {"eta": 0.3, "max_depth": 6}
N_BOOST = 100
SEED = 0
DATASETS = ["hill-valley", "arrhythmia", "semeion", "libras"]


def greedy_oracle(sp):
    m = FeatXGBoost(n_boost=N_BOOST, xgb_cfg=CFG, seed=SEED)
    ytr = sp.y_train.astype(int)
    m._ytr = ytr
    m.K = int(ytr.max() + 1)
    m._precompute(sp.X_train, sp.y_train, sp.X_val, sp.X_test)
    yval = sp.y_val.astype(int)
    N = len(ytr)
    onehot = np.eye(m.K)[ytr]
    Fm = np.zeros((N, m.K))
    Fmv = np.zeros((len(yval), m.K))
    Fmt = np.zeros((len(sp.y_test), m.K))
    rng = np.random.default_rng(SEED)
    base = {"objective": "multi:softprob", "num_class": m.K,
            "verbosity": 0, "seed": SEED, **CFG}
    hist, cost = [], 0.0
    for step in range(N_BOOST):
        g = _softmax(Fm) - onehot
        idx, w = goss_sample(np.linalg.norm(g, axis=1), a=m.a, b=m.b, rng=rng)
        best = None
        for fid, name in enumerate(m.fe_names):
            Z = m.Z["train"][name]
            dtr = xgb.DMatrix(Z[idx], label=ytr[idx], weight=w,
                              base_margin=Fm[idx].ravel())
            bst = xgb.train(base, dtr, num_boost_round=1)
            cval = bst.predict(m._dmat["val"][name], output_margin=True).reshape(-1, m.K)
            loss = _logloss(_softmax(Fmv + cval), yval)
            if best is None or loss < best[0]:
                ctr = bst.predict(m._dmat["train"][name], output_margin=True).reshape(-1, m.K)
                ctt = bst.predict(m._dmat["test"][name], output_margin=True).reshape(-1, m.K)
                best = (loss, fid, ctr, cval, ctt)
        _, fid, ctr, cval, ctt = best
        Fm += ctr; Fmv += cval; Fmt += ctt
        hist.append(fid); cost += m.fe_cost[fid]
    acc = (np.argmax(Fmt, axis=1) == sp.y_test).mean()
    return acc, cost, hist, m.fe_names


def ref_acc(sp, selector):
    m = FeatXGBoost(selector=selector, n_boost=N_BOOST, xgb_cfg=CFG,
                    lam=0.0, warmup=0, seed=SEED)
    m.fit(sp.X_train, sp.y_train, sp.X_val, sp.y_val, sp.X_test)
    return (m.predict("test") == sp.y_test).mean()


def main():
    ht = FE_ARMS.index("ht_svd")
    for name in DATASETS:
        sp = prepare(UCIDataset(name).get_split(0))
        o_acc, o_cost, hist, names = greedy_oracle(sp)
        fx = ref_acc(sp, FixedSelector(ht))
        rr = ref_acc(sp, RoundRobinSelector(len(names)))
        hist = np.array(hist)
        print(f"\n===== {name} (fold 0) =====")
        print(f"  Round-robin={rr:.4f}  Fixed-ht_svd={fx:.4f}  "
              f"ORACLE={o_acc:.4f}  (oracle−fixHT = {o_acc-fx:+.4f})")
        # phân bố FE oracle theo 3 giai đoạn
        thirds = np.array_split(hist, 3)
        print("  Phân bố FE oracle theo giai đoạn boosting:")
        for lbl, part in zip(["early", "mid  ", "late "], thirds):
            c = np.bincount(part, minlength=len(names))
            top = sorted(zip(names, c), key=lambda kv: -kv[1])[:3]
            line = ", ".join(f"{k}={v}" for k, v in top if v > 0)
            print(f"    {lbl}: {line}")


if __name__ == "__main__":
    main()
