"""
Claim 2 — Động học FE theo bước boosting (dữ liệu cho 'timeline graph').

Chạy greedy oracle, log phép FE tốt nhất ở TỪNG bước, xuất CSV để vẽ:
  results/oracle_timeline.csv       : (dataset, fold, step, fe)  — chi tiết từng bước
  results/oracle_phase_summary.csv  : (dataset, phase, fe, count) — gộp early/mid/late

Bỏ 3 tập plant-* (100 lớp) vì oracle 6× quá chậm; dùng 12 tập còn lại.
CHẠY SAU KHI study.py xong (tránh giành CPU).
"""

import csv
from pathlib import Path

import numpy as np
import xgboost as xgb

from src.fe.registry import FE_ARMS
from src.models.feat_xgboost import FeatXGBoost, _logloss, _softmax
from src.goss import goss_sample
from src.preprocessing import prepare
from src.data_loader import UCIDataset
from src.select_datasets import hard_set

CFG = {"eta": 0.3, "max_depth": 6}
N_BOOST = 100
SEED = 0
FOLDS = [0, 1, 2, 3]
SKIP = {"plant-shape", "plant-margin", "plant-texture"}  # K=100, oracle quá chậm
OUT = Path(__file__).resolve().parent.parent / "results"


def greedy_oracle_hist(sp):
    m = FeatXGBoost(n_boost=N_BOOST, xgb_cfg=CFG, seed=SEED)
    ytr = sp.y_train.astype(int)
    m._ytr = ytr
    m.K = int(ytr.max() + 1)
    m._precompute(sp.X_train, sp.y_train, sp.X_val, sp.X_test)
    yval = sp.y_val.astype(int)
    onehot = np.eye(m.K)[ytr]
    Fm = np.zeros((len(ytr), m.K))
    Fmv = np.zeros((len(yval), m.K))
    rng = np.random.default_rng(SEED)
    base = {"objective": "multi:softprob", "num_class": m.K,
            "verbosity": 0, "seed": SEED, **CFG}
    hist = []
    for _ in range(N_BOOST):
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
                best = (loss, fid, ctr, cval)
        _, fid, ctr, cval = best
        Fm += ctr; Fmv += cval
        hist.append(fid)
    return hist


def main():
    OUT.mkdir(exist_ok=True)
    datasets = [d for d in hard_set() if d not in SKIP]
    detail, summary = [], []
    for name in datasets:
        ds = UCIDataset(name)
        for f in FOLDS:
            sp = prepare(ds.get_split(f))
            hist = greedy_oracle_hist(sp)
            for step, fid in enumerate(hist):
                detail.append([name, f, step, FE_ARMS[fid]])
            # gộp 3 giai đoạn
            for lbl, part in zip(["early", "mid", "late"], np.array_split(hist, 3)):
                c = np.bincount(part, minlength=len(FE_ARMS))
                for fid, cnt in enumerate(c):
                    summary.append([name, f, lbl, FE_ARMS[fid], int(cnt)])
        print(f"{name}: xong 4 fold", flush=True)

    with open(OUT / "oracle_timeline.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["dataset", "fold", "step", "fe"]); w.writerows(detail)
    with open(OUT / "oracle_phase_summary.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["dataset", "fold", "phase", "fe", "count"]); w.writerows(summary)

    # in tổng hợp early vs late toàn bộ
    print("\n=== Tần suất FE early vs late (gộp mọi dataset/fold) ===")
    import collections
    agg = collections.defaultdict(lambda: collections.Counter())
    for name, f, phase, fe, cnt in summary:
        agg[phase][fe] += cnt
    for phase in ["early", "mid", "late"]:
        tot = sum(agg[phase].values())
        top = agg[phase].most_common(4)
        print(f"  {phase:5s}: " + ", ".join(f"{fe}={c/tot*100:.0f}%" for fe, c in top))


if __name__ == "__main__":
    main()
