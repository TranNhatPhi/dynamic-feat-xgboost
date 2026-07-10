"""
Hình 1 — Biểu đồ đánh đổi Accuracy vs Cost (memory thật), có đường biên Pareto.

Cho thấy: bandit bị cheap-rotation THỐNG TRỊ (nằm dưới-phải đường biên), và cheap-rotation
là 'điểm knee' (gần accuracy cao nhất ở ~⅓ chi phí của round-robin).
Đọc số từ results/table1_accuracy.csv (accuracy TB) + results/benchmark.csv (memory TB).
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results"

# nhãn đẹp + màu (colorblind-safe)
META = {
    "plain_xgb":   ("Plain XGBoost (0 FE)", "#888888", "o"),
    "fixed_htsvd": ("Fixed HT-SVD (1 FE)",  "#0072B2", "s"),
    "twofe":       ("2-FE rotation",        "#56B4E9", "^"),
    "cheap_rr":    ("Cheap rotation (5 FE)", "#009E73", "D"),
    "round_robin": ("Round-robin (6 FE, gốc)", "#E69F00", "P"),
    "bandit":      ("Contextual bandit",    "#D55E00", "X"),
}


def col_means(path, valcol):
    d = defaultdict(list)
    for r in csv.DictReader(open(path)):
        d[r["config"]].append(float(r[valcol]))
    return {k: np.mean(v) for k, v in d.items()}


def main():
    # accuracy TB mỗi config (trung bình trên 15 dataset)
    acc_rows = list(csv.reader(open(OUT / "table1_accuracy.csv")))
    header = acc_rows[0]
    acc = {c: np.mean([float(row[header.index(c)]) for row in acc_rows[1:]])
           for c in META if c in header}
    mem = col_means(OUT / "benchmark.csv", "fe_mem_mb")

    configs = [c for c in META if c in acc and c in mem]
    xs = np.array([mem[c] for c in configs])
    ys = np.array([acc[c] for c in configs])

    # đường biên Pareto (muốn x nhỏ, y lớn): sắp theo x tăng, giữ điểm có y cao dần
    order = np.argsort(xs)
    pareto, best_y = [], -1
    for i in order:
        if ys[i] > best_y + 1e-9:
            pareto.append(i); best_y = ys[i]
    pareto_x = xs[pareto]; pareto_y = ys[pareto]

    fig, ax = plt.subplots(figsize=(7.2, 5.0), dpi=200)
    # đường biên
    ax.plot(pareto_x, pareto_y, "--", color="#444", lw=1.2, zorder=1,
            label="Đường biên Pareto")
    # các điểm
    for c, x, y in zip(configs, xs, ys):
        label, color, mk = META[c]
        ax.scatter(x, y, s=170, c=color, marker=mk, edgecolors="white",
                   linewidths=1.2, zorder=3, label=label)

    # chú thích 2 điểm quan trọng
    ax.annotate("Điểm 'knee' —\ntốt nhất về đánh đổi",
                xy=(mem["cheap_rr"], acc["cheap_rr"]),
                xytext=(mem["cheap_rr"] - 0.2, acc["cheap_rr"] - 0.011),
                fontsize=9, color="#009E73", ha="center",
                arrowprops=dict(arrowstyle="->", color="#009E73"))
    ax.annotate("BỊ THỐNG TRỊ\n(cùng RAM round-robin,\naccuracy thấp hơn cheap)",
                xy=(mem["bandit"], acc["bandit"]),
                xytext=(mem["bandit"] - 1.7, acc["bandit"] - 0.012),
                fontsize=9, color="#D55E00", ha="center",
                arrowprops=dict(arrowstyle="->", color="#D55E00"))

    ax.set_xlabel("Chi phí bộ nhớ FE thật (MB) — thấp hơn tốt hơn →", fontsize=11)
    ax.set_ylabel("Accuracy trung bình (15 tập)", fontsize=11)
    ax.set_title("Đánh đổi Accuracy – Chi phí của các chiến lược chọn FE\n"
                 "(embedded feature engineering trong XGBoost boosting)", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(ys.min() - 0.004, ys.max() + 0.004)  # hiện đủ cả điểm plain
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    ax.invert_xaxis()  # chi phí thấp bên phải để "tốt = phải-trên"
    fig.tight_layout()
    fig.savefig(OUT / "fig1_pareto.png", bbox_inches="tight")
    print("Đã lưu results/fig1_pareto.png")
    print("\nToạ độ (mem MB, acc):")
    for c in configs:
        star = "  <-- Pareto" if configs.index(c) in pareto else ""
        print(f"  {META[c][0]:26s} ({mem[c]:.2f}, {acc[c]:.4f}){star}")


if __name__ == "__main__":
    main()
