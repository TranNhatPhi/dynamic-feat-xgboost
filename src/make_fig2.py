"""Hình 2 — Phân bố phép FE 'tối ưu' (oracle) theo giai đoạn boosting (RQ4)."""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent.parent / "results"
FE_ORDER = ["identity", "autofeat", "random_proj", "ht_svd", "robust", "minmax"]
COLORS = {"identity": "#888888", "autofeat": "#D55E00", "random_proj": "#56B4E9",
          "ht_svd": "#009E73", "robust": "#CC79A7", "minmax": "#E69F00"}
PHASES = ["early", "mid", "late"]


def main():
    counts = defaultdict(lambda: defaultdict(float))  # phase -> fe -> count
    for r in csv.DictReader(open(OUT / "oracle_phase_summary.csv")):
        counts[r["phase"]][r["fe"]] += float(r["count"])
    # chuẩn hoá theo phase
    frac = {p: {fe: counts[p].get(fe, 0) / max(sum(counts[p].values()), 1)
                for fe in FE_ORDER} for p in PHASES}

    x = np.arange(len(PHASES))
    w = 0.13
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=200)
    for i, fe in enumerate(FE_ORDER):
        vals = [frac[p][fe] * 100 for p in PHASES]
        ax.bar(x + (i - 2.5) * w, vals, w, label=fe, color=COLORS[fe],
               edgecolor="white", linewidth=0.5)
    ax.set_xticks(x); ax.set_xticklabels(["Early\n(bước 1–33)", "Mid\n(34–66)", "Late\n(67–100)"])
    ax.set_ylabel("Tần suất được oracle chọn (%)")
    ax.set_title("Phân bố phép FE tối ưu (oracle) theo giai đoạn boosting\n"
                 "→ gần như phẳng: không có quy luật thời gian mạnh để học online (RQ4)")
    ax.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14))
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "fig2_oracle_phase.png", bbox_inches="tight")
    print("Đã lưu results/fig2_oracle_phase.png")
    for p in PHASES:
        print(f"  {p:5s}: " + ", ".join(f"{fe}={frac[p][fe]*100:.0f}%" for fe in FE_ORDER))


if __name__ == "__main__":
    main()
