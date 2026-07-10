"""
Gộp Bảng 1 tổng hợp từ mọi CSV thí nghiệm (study / cheap_rr / twofe).

Sắp theo "thang đa dạng FE": plain(0) → fixed(1) → twofe(2) → cheap(5) → round(6) → bandit(adaptive).
In accuracy + cost mỗi dataset, trung bình toàn cục, và các paired t-test then chốt.
"""

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results"
FILES = {
    "study_long.csv": ["plain_xgb", "round_robin", "fixed_htsvd", "bandit"],
    "cheap_rr_long.csv": ["cheap_rr"],
    "twofe_long.csv": ["twofe"],
}
# thứ tự cột theo thang đa dạng
ORDER = ["plain_xgb", "fixed_htsvd", "twofe", "cheap_rr", "round_robin", "bandit"]
LABEL = {"plain_xgb": "plain(0)", "fixed_htsvd": "fixed(1)", "twofe": "2FE",
         "cheap_rr": "cheap(5)", "round_robin": "round(6)", "bandit": "bandit"}


def load():
    acc = defaultdict(lambda: defaultdict(list))   # config -> dataset -> [acc]
    cost = defaultdict(lambda: defaultdict(list))
    for fname, keep in FILES.items():
        p = OUT / fname
        if not p.exists():
            continue
        for r in csv.DictReader(open(p)):
            c = r["config"]
            if c in keep:
                acc[c][r["dataset"]].append(float(r["acc"]))
                cost[c][r["dataset"]].append(float(r["cost"]))
    return acc, cost


def paired(a, b):
    d = np.array(a) - np.array(b)
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else float("inf")
    return d.mean(), d.std(ddof=1), t, int((d > 1e-4).sum()), int((np.abs(d) <= 1e-4).sum()), int((d < -1e-4).sum())


def main():
    acc, cost = load()
    datasets = sorted(acc["bandit"].keys())

    # ---- Bảng 1: accuracy ----
    print("BẢNG 1 — Accuracy trung bình (4 fold × 3 seed)\n")
    hdr = "dataset".ljust(28) + "".join(LABEL[c].rjust(10) for c in ORDER)
    print(hdr); print("-" * len(hdr))
    rows_csv = [["dataset"] + ORDER]
    for ds in datasets:
        line = ds.ljust(28)
        row = [ds]
        for c in ORDER:
            v = np.mean(acc[c][ds]) if acc[c][ds] else float("nan")
            line += f"{v:10.3f}"; row.append(round(v, 4))
        print(line); rows_csv.append(row)

    # trung bình toàn cục
    print("-" * len(hdr))
    line = "TRUNG BÌNH".ljust(28)
    for c in ORDER:
        vals = [np.mean(acc[c][ds]) for ds in datasets if acc[c][ds]]
        line += f"{np.mean(vals):10.3f}"
    print(line)

    # cost trung bình
    line = "COST trung bình".ljust(28)
    for c in ORDER:
        vals = [np.mean(cost[c][ds]) for ds in datasets if cost[c][ds]]
        line += f"{np.mean(vals):10.1f}"
    print(line)

    with open(OUT / "table1_accuracy.csv", "w", newline="") as f:
        csv.writer(f).writerows(rows_csv)

    # ---- paired t-tests then chốt ----
    def vec(c): return [np.mean(acc[c][ds]) for ds in datasets]
    print("\nCÁC KIỂM ĐỊNH GHÉP CẶP (n=%d):" % len(datasets))
    for a, b in [("round_robin", "plain_xgb"), ("round_robin", "fixed_htsvd"),
                 ("cheap_rr", "round_robin"), ("bandit", "cheap_rr"),
                 ("bandit", "twofe"), ("bandit", "fixed_htsvd")]:
        m, sd, t, w, ti, l = paired(vec(a), vec(b))
        print(f"  {a:12s} vs {b:12s}: Δ={m:+.4f}±{sd:.4f} t={t:+.2f} (W/T/L={w}/{ti}/{l})")
    print("\nĐã ghi results/table1_accuracy.csv")


if __name__ == "__main__":
    main()
