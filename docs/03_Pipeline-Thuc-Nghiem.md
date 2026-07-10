# 03 — Pipeline thực nghiệm (từ data đến kết quả)

## Sơ đồ tổng thể

```
data/raw/<tên>/         (đã có)
      │  data_loader.py
      ▼
 Split(train/val/test)   (đã có)
      │
      ▼
 [Tiền xử lý tối thiểu]  encode nhãn 0..K-1 nếu cần
      │
      ▼
 ┌─────────────────────────────────────────────┐
 │ MÔ HÌNH (mỗi bước boosting)                  │
 │   GOSS → chọn FE (dynamic/all/fixed) → dựng cây│
 └─────────────────────────────────────────────┘
      │
      ▼
 Đánh giá: accuracy, F1(macro/weighted), precision, recall
      │
      ▼
 Benchmark: wall-clock time + peak memory (đo song song)
      │
      ▼
 results/*.csv  →  bảng & biểu đồ  →  paper
```

## Cấu trúc thư mục code sẽ xây

```
src/
  data_loader.py          [đã có]
  preprocessing.py        encode nhãn, kiểm tra sạch
  fe/                     các phép feature engineering
    __init__.py
    identity.py
    random_projection.py
    ht_svd.py
    scalers.py            robust + minmax
    autofeat_wrap.py
    registry.py           map tên -> hàm FE (để MAB chọn)
  goss.py                 lấy mẫu GOSS (a=0.45, b=0.35)
  models/
    feat_xgboost.py       Feat-XGBoost (nhúng FE vào boosting)
    dynamic_feat.py       phiên bản MAB chọn động  [NOVELTY]
    bandit.py             UCB1 / epsilon-greedy
  tuning.py               Optuna (100 trial như bài gốc)
  benchmark.py            đo time + peak memory (psutil/tracemalloc)
  run_experiment.py       CLI: chạy 1 hoặc nhiều dataset, xuất CSV
  config.yaml             tham số (a, b, n_estimators, danh sách dataset...)
```

## Các phương pháp cần chạy (cột trong bảng kết quả)

1. **XGBoost gốc** (baseline chính, tự chạy để lấy mốc runtime/memory).
2. **Feat-XGBoost (tái hiện)** — cấu hình All-FE như bài gốc.
3. **Fixed-cheap-FE**, **Random-FE** (ablation).
4. **Dynamic-FE (của mình)** — điểm chính.

## Chỉ số đánh giá

- **Chất lượng:** accuracy, F1-macro, F1-weighted, precision, recall (khớp file gốc
  `results/original/results_*.csv`).
- **Hiệu năng:** thời gian train (wall-clock, `time.perf_counter`), peak RAM
  (`tracemalloc`/`psutil`), (tùy chọn) số phép FE thực thi.
- Mỗi cấu hình chạy **≥3 lần**, báo cáo **mean ± std**.

## Giao thức đánh giá (giữ đúng bài gốc)

- 4-fold CV theo split có sẵn; tune siêu tham số bằng **Optuna 100 trial** trên validation.
- Chốt siêu tham số theo validation, báo cáo trên **test** (không nhìn test khi tune).
- Cùng seed, cùng máy cho mọi phương pháp khi so runtime/memory.

## Kế hoạch chạy theo giai đoạn (tiết kiệm tiền)

| GĐ | Máy | Việc | Mục tiêu |
|---|---|---|---|
| 1 | Mac M4 | 3-5 dataset nhỏ (wine, iris, seeds...) | debug pipeline chạy đúng, đủ 4 phương pháp |
| 2 | Mac M4 (đêm) | ~20 dataset vừa | tái hiện Feat-XGBoost, so accuracy với bảng gốc |
| 3 | Cloud (nếu cần) | full 61 dataset | benchmark hoàn chỉnh + ablation |
| 4 | Mac | phân tích + vẽ hình | bảng, biểu đồ cho paper |

> **Nguyên tắc vàng:** KHÔNG lên cloud khi pipeline chưa chạy đúng trên Mac. Debug trên
> cloud = đốt tiền theo giờ.

## Lệnh dự kiến (khi code xong)

```bash
# chạy thử 1 dataset, 1 phương pháp
./.venv/bin/python -m src.run_experiment --dataset wine --method dynamic --folds 0

# chạy nhiều dataset, mọi phương pháp, xuất CSV
./.venv/bin/python -m src.run_experiment \
    --datasets wine,iris,seeds \
    --methods xgb,featxgb,fixed,random,dynamic \
    --repeats 3 --out results/run1.csv
```

## Việc tiếp theo (thứ tự code)

1. `preprocessing.py` + `fe/` (các phép FE, có `registry`).
2. `goss.py`.
3. `models/feat_xgboost.py` (tái hiện bài gốc trước — để có mốc đúng).
4. `benchmark.py` + `run_experiment.py` (chạy được end-to-end trên wine).
5. `models/dynamic_feat.py` + `bandit.py` (novelty).
6. `tuning.py` (Optuna) — thêm sau khi pipeline ổn.
