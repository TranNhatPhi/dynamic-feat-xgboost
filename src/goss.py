"""
GOSS — Gradient-based One-Side Sampling (như LightGBM / bài gốc).

Ở mỗi bước boosting: giữ TOÀN BỘ mẫu gradient lớn (mẫu "khó", top a%), lấy NGẪU NHIÊN
một phần b% trong số mẫu gradient nhỏ (mẫu "dễ"), và NHÂN TRỌNG SỐ nhóm nhỏ này với
(1-a)/b để BẢO TOÀN phân phối gốc khi tính information gain.

Tham số bài gốc: a = 0.45, b = 0.35.
"""

from __future__ import annotations

import numpy as np


def goss_sample(
    gradients: np.ndarray,
    a: float = 0.45,
    b: float = 0.35,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Trả về (indices, weights):
      - indices : chỉ số mẫu được chọn (top-gradient + phần nhỏ ngẫu nhiên)
      - weights : trọng số tương ứng (nhóm gradient nhỏ được khuếch đại (1-a)/b)
    """
    if rng is None:
        rng = np.random.default_rng()
    if not (0 < a < 1 and 0 < b < 1 and a + b <= 1):
        raise ValueError(f"Cần 0<a<1, 0<b<1, a+b<=1; nhận a={a}, b={b}")

    g = np.abs(np.asarray(gradients, dtype=float)).reshape(-1)
    n = g.shape[0]
    n_top = int(a * n)
    n_rand = int(b * n)

    order = np.argsort(-g)                 # sắp giảm dần theo |gradient|
    top_idx = order[:n_top]                # mẫu khó — giữ hết
    rest = order[n_top:]                   # mẫu dễ

    if rest.size > 0 and n_rand > 0:
        rand_idx = rng.choice(rest, size=min(n_rand, rest.size), replace=False)
    else:
        rand_idx = np.empty(0, dtype=np.int64)

    idx = np.concatenate([top_idx, rand_idx]).astype(np.int64)
    weights = np.ones(idx.shape[0], dtype=np.float64)
    amp = (1.0 - a) / b if b > 0 else 1.0
    weights[len(top_idx):] = amp           # khuếch đại nhóm gradient nhỏ
    return idx, weights


def hard_sample_stats(gradients: np.ndarray, idx: np.ndarray) -> dict:
    """Vài thống kê nhanh về tập được chọn (phục vụ log/debug)."""
    g = np.abs(np.asarray(gradients, dtype=float)).reshape(-1)
    return {
        "n_total": int(g.size),
        "n_selected": int(idx.size),
        "frac": float(idx.size / max(g.size, 1)),
        "mean_grad_selected": float(g[idx].mean()) if idx.size else 0.0,
        "mean_grad_all": float(g.mean()),
    }


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    grads = rng.normal(size=2000)
    idx, w = goss_sample(grads, a=0.45, b=0.35, rng=rng)
    st = hard_sample_stats(grads, idx)
    print("GOSS a=0.45 b=0.35 trên 2000 mẫu:")
    print(f"  chọn {st['n_selected']}/{st['n_total']} = {st['frac']*100:.1f}% "
          f"(kỳ vọng ~80%)")
    print(f"  |grad| TB nhóm chọn = {st['mean_grad_selected']:.3f} "
          f"vs toàn bộ = {st['mean_grad_all']:.3f}  (chọn phải lớn hơn)")
    print(f"  trọng số khuếch đại nhóm dễ = {(1-0.45)/0.35:.3f}")
    print(f"  số trọng số >1: {(w>1).sum()} (nhóm gradient nhỏ)")
