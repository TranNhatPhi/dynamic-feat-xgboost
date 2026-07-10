# 01 — Quy trình xử lý dữ liệu

## Nguồn dữ liệu

- **Bộ benchmark UCI 121 datasets** (Klambauer / Fernández-Delgado), đã được tác giả bài
  báo chia split chuẩn và public tại:
  https://github.com/lingping-fuzzy/UCI-data-correct-split
- Đây là bản **đã sửa lỗi split**. Tác giả chỉ ra nhiều nghiên cứu trước dùng split lỗi
  (ví dụ train có 6 nhãn nhưng test chỉ 5 nhãn) → kết quả sai lệch. Phải dùng bản này.

## Định dạng dữ liệu (đã được số hoá sẵn)

Mỗi dataset nằm trong `data/raw/<tên>/`, gồm 4 file text, **delimiter = dấu phẩy `,`**:

| File | Ý nghĩa | Kích thước |
|---|---|---|
| `<tên>_py.dat` | ma trận đặc trưng X | n_mẫu × n_features |
| `labels_py.dat` | nhãn y (đã mã hoá 0..K-1) | n_mẫu |
| `folds_py.dat` | chỉ số fold | n_mẫu × 4 |
| `validation_folds_py.dat` | chỉ số validation | n_mẫu × 4 |

> **Quan trọng:** "xử lý data thô" ở đây KHÔNG phải làm sạch/encode lại. Data đã số hoá.
> Việc scale (RobustScaler/MinMaxScaler) là một phần của **feature engineering lúc train**,
> không làm ở bước load. Làm sai chỗ này là hỏng so sánh với bài gốc.

## Logic chia fold (4-fold CV)

Với mỗi fold `cv ∈ {0,1,2,3}` (cột thứ `cv` của ma trận folds/validation):

```
test  = các dòng có folds[:, cv] == 1
val   = các dòng có validation[:, cv] == 1
train = các dòng có folds[:, cv] == 0  VÀ  validation[:, cv] == 0
```

(Sao chép chính xác từ `reference_code/featureEng/datasets/UCIdata.py` của tác giả,
nhưng bỏ phụ thuộc `torch`.)

## Các bước đã làm

1. `git clone` repo data → giải nén `part1..part3` + `part4 (40 problem-corrected)`.
2. Gộp phẳng tất cả về `data/raw/<tên>/` (81 từ part1-3 + 40 corrected = **121**, không trùng tên).
3. Viết `src/data_loader.py`.
4. **Kiểm chứng** khớp `data/meta_datasets.csv`:
   - wine: 100/34/44 (train/val/test) ✓
   - iris: 92/21/37 ✓
   - balloons: 9/3/4 (đúng bản "new split" corrected) ✓

## Cách dùng loader

```python
from src.data_loader import UCIDataset, list_datasets, load

# liệt kê tất cả dataset
names = list_datasets()          # -> 121 tên

# nạp một dataset, một fold
ds = UCIDataset('wine')
print(ds.X.shape, ds.n_classes, ds.n_folds)   # (178,13) 3 4
s = ds.get_split(0)              # fold 0
s.X_train, s.y_train, s.X_val, s.y_val, s.X_test, s.y_test

# tiện ích ngắn
s = load('iris', cv=2)
```

Chạy kiểm tra nhanh:

```bash
./.venv/bin/python src/data_loader.py
```

## Preprocessing (ĐÃ XONG — `src/preprocessing.py`)

- `prepare(split)`: remap nhãn về `0..K-1` (union của train/val/test), ép X về float32
  sạch. KHÔNG scale (scale thuộc FE lúc train).
- `check(split)`: phát hiện nhãn không liên tục, NaN/Inf, và **class thiếu trong train**.
- `scan_all()`: quét toàn bộ. Chạy: `./.venv/bin/python -m src.preprocessing`.

**Kết quả quét (121 × 4 fold = 484): 483 OK.** Riêng **`low-res-spect`** có fold thiếu
class 5 trong train (đúng loại bug bài báo cảnh báo) → **LOẠI khỏi thí nghiệm**, ghi rõ trong paper.

## Việc cần chốt sau

- **Danh sách 61 dataset** dùng cho thí nghiệm (bài gốc dùng 61 trong tổng 121).
  Xem `reference_code/featureEng/datasets/UCIdata.py` — biến `label_to_name` (đang bật 44)
  + phần comment (57 nữa). Cần đối chiếu với bảng kết quả trong paper để lấy đúng 61.
  Nhớ trừ `low-res-spect`.
