# Dynamic Feat-XGBoost (NCKH — GVHD: thầy Lý Quang Vinh)

Dự án cải tiến bài báo **"Enhancing sampling performance in XGBoost by ensemble
feature engineering"** (Feat-XGBoost / Mix-XGBoost, *Pattern Recognition* 2026).

- Repo code gốc: https://github.com/lingping-fuzzy/XGBoost-by-ensemble-feature-engineering
- Repo data (split chuẩn): https://github.com/lingping-fuzzy/UCI-data-correct-split

## Mục tiêu

Giữ độ chính xác tương đương bài gốc nhưng **giảm thời gian chạy & bộ nhớ**, và
thêm tính mới: chọn động (dynamic) kỹ thuật feature engineering trong từng bước
boosting thay vì áp dụng cả cụm. Nhắm hội nghị/tạp chí Q4 (chắc tay trước).

## Cấu trúc

```
data/
  meta_datasets.csv     thông tin 121 dataset (số mẫu, lớp, feature, split size)
  raw/<ten>/            121 dataset UCI, mỗi cái 4 file .dat (delimiter = ',')
src/
  data_loader.py        bộ nạp data (numpy thuần), sao chép logic split của tác giả
reference_code/         code gốc của tác giả (chỉ để tham khảo, KHÔNG sửa)
results/original/       kết quả gốc từ bài báo (acc, f1, precision, recall...)
.venv/                  môi trường Python
```

## Định dạng dữ liệu (quan trọng)

Data đã được số hoá & chia sẵn (bộ benchmark UCI 121 của Klambauer/Fernández-Delgado).
Bước "xử lý data thô" ở đây = nạp đúng file + chia đúng fold, KHÔNG scale/clean lại
(scale nằm trong tập feature engineering lúc train). Mỗi dataset có 4-fold CV:

- `test`  = nơi `folds_py.dat[:,cv] == 1`
- `val`   = nơi `validation_folds_py.dat[:,cv] == 1`
- `train` = phần còn lại (`folds==0 & validation==0`)

## Chạy thử

```bash
./.venv/bin/python src/data_loader.py     # in ra 121 dataset + shape các fold
```

Đã kiểm chứng khớp với `data/meta_datasets.csv` (wine 100/34/44, iris 92/21/37...).

## Việc tiếp theo (chưa làm)

1. Cài `scikit-learn`, `xgboost`, `autofeat` vào .venv.
2. Chốt danh sách dataset dùng để thí nghiệm (bài gốc dùng 61).
3. Chạy lại **Feat-XGBoost gốc** trên máy này để lấy mốc so sánh runtime/memory
   (BẮT BUỘC — không được copy runtime từ bài báo).
4. Hiện thực cơ chế chọn động FE + ablation (fixed-FE / random-FE / dynamic).
