# Hướng dẫn chạy Dynamic Feat-XGBoost trên Vast.ai (H100)

> Tài liệu này hướng dẫn đóng gói dự án ở máy Mac M4, thuê một máy H100 trên
> Vast.ai, đẩy code+data lên, cài môi trường và chạy. Kèm lưu ý về cách đo
> runtime/memory cho **hợp lệ** khi so sánh với bài báo gốc.

---

## ⚠️ Đọc trước — sự thật về H100 cho bài này

XGBoost trên dữ liệu bảng UCI (đa số < vài chục nghìn dòng) là bài toán **CPU-bound**.
Phần lớn pipeline (autofeat, MDS, t-SNE, Isomap, random projection, HT-SVD) chạy
**CPU (sklearn/numpy)**, GPU H100 KHÔNG tăng tốc được. GPU chỉ giúp phần `XGBoost.fit`
khi đặt `device='cuda'`, và ngay cả vậy với data nhỏ thường **không nhanh hơn** CPU
nhiều nhân do chi phí copy lên VRAM.

→ Nếu vẫn dùng H100: **tận dụng luôn nhiều nhân CPU của máy đó** (H100 thường đi kèm
CPU mạnh) để song song hoá, và chỉ bật `device='cuda'` cho các dataset lớn
(miniboone, connect-4, statlog-shuttle, adult). Đừng để trả tiền GPU mà GPU nằm chơi.

---

## Bước 0 — Đóng gói dự án ở máy Mac

Data (`data/raw`, ~40MB) không nên commit git. Ta nén cả project (trừ `.venv`) thành 1 file:

```bash
cd /Users/trannhatphi/Documents/NCKH-thayVinh
tar --exclude='.venv' --exclude='.git' --exclude='.DS_Store' \
    -czf ~/featxgb_bundle.tar.gz .
ls -lh ~/featxgb_bundle.tar.gz     # kiểm tra kích thước (~40-60MB)
```

## Bước 1 — Thuê máy H100 trên Vast.ai

1. Vào https://vast.ai → **Console → Search**.
2. Chọn image mẫu: **`pytorch/pytorch`** hoặc **`nvidia/cuda:12.4.1-runtime-ubuntu22.04`**
   (có sẵn CUDA driver — XGBoost bản pip tự dùng được).
3. Bộ lọc gợi ý:
   - GPU: `H100` (1 cái là đủ, đừng thuê nhiều).
   - **CPU cores ≥ 16**, **RAM ≥ 64 GB** (quan trọng hơn GPU cho bài này!).
   - **Disk ≥ 30 GB**.
4. **Rent** → vào tab **Instances**, bấm nút **`>_`** (hoặc copy dòng lệnh SSH).
   Vast cho em thông tin dạng: `ssh -p <PORT> root@<HOST>`.

## Bước 2 — Đẩy bundle lên máy remote

Từ máy Mac (thay `<PORT>` `<HOST>` bằng của em):

```bash
scp -P <PORT> ~/featxgb_bundle.tar.gz root@<HOST>:/workspace/
```

Rồi SSH vào:

```bash
ssh -p <PORT> root@<HOST>
cd /workspace && mkdir -p featxgb && tar -xzf featxgb_bundle.tar.gz -C featxgb && cd featxgb
```

## Bước 3 — Cài môi trường trên remote (Linux)

```bash
apt-get update -y && apt-get install -y python3-venv python3-pip   # nếu thiếu
python3 -m venv .venv
./.venv/bin/pip install --upgrade pip
./.venv/bin/pip install -r requirements.txt

# Kiểm tra XGBoost thấy GPU:
./.venv/bin/python -c "import xgboost as xgb; print('xgboost', xgb.__version__)"
nvidia-smi        # phải thấy H100
```

> Nếu `autofeat` cài lỗi (hay kén phiên bản), tạm bỏ nó khỏi requirements để chạy được
> phần còn lại, rồi cài riêng sau: `./.venv/bin/pip install autofeat`.

## Bước 4 — Kiểm tra data & loader chạy được

```bash
./.venv/bin/python src/data_loader.py     # phải in ra "Tìm thấy 121 dataset..."
```

## Bước 5 — Chạy thử XGBoost trên GPU (1 dataset lớn)

Đây là cách bật GPU đúng trong XGBoost 2.x — dùng `device='cuda'`, `tree_method='hist'`:

```bash
./.venv/bin/python - <<'PY'
import time
from src.data_loader import UCIDataset
from xgboost import XGBClassifier

for name in ['letter', 'miniboone']:
    d = UCIDataset(name); s = d.get_split(0)
    for dev in ['cpu', 'cuda']:
        clf = XGBClassifier(n_estimators=500, max_depth=6,
                            tree_method='hist', device=dev, n_jobs=-1, verbosity=0)
        t = time.time(); clf.fit(s.X_train, s.y_train); dt = time.time() - t
        acc = (clf.predict(s.X_test) == s.y_test).mean()
        print(f'{name:12s} device={dev:4s} train={dt:6.2f}s acc={acc:.4f}')
PY
```

So sánh dòng `cpu` vs `cuda` — nếu `cuda` không nhanh hơn rõ rệt thì cứ dùng CPU.

## Bước 6 — Đo runtime & memory CHO HỢP LỆ (rất quan trọng cho paper)

Vì đóng góp của em là "nhanh hơn / ít RAM hơn", cách đo phải chặt:

- **Cùng một máy** cho cả phương pháp gốc và phương pháp mới (đừng so máy này với số trong bài báo).
- **Wall-clock time**: `time.perf_counter()` bao quanh đúng đoạn train (không tính load data).
- **Peak memory**: dùng `psutil` hoặc `tracemalloc`:
  ```python
  import tracemalloc; tracemalloc.start()
  # ... train ...
  cur, peak = tracemalloc.get_traced_memory(); print('peak MB', peak/1e6)
  ```
- **Chạy lặp lại** mỗi cấu hình 3-5 lần, báo cáo trung bình ± độ lệch chuẩn.
- Ghi rõ cấu hình máy (CPU, RAM, GPU) trong phần Experiments của paper.

## Bước 7 — Kéo kết quả về máy Mac

```bash
# trên máy Mac:
scp -P <PORT> root@<HOST>:/workspace/featxgb/results/*.csv \
    /Users/trannhatphi/Documents/NCKH-thayVinh/results/
```

## Bước 8 — TẮT MÁY để khỏi cháy túi 💸

Vast tính tiền theo giờ kể cả khi không chạy. Xong việc:
**Console → Instances → Destroy** (hoặc **Stop** nếu muốn giữ data, vẫn tốn phí disk).
Nhớ kéo kết quả về trước khi Destroy.

---

## Checklist nhanh

- [ ] Đóng gói bundle (Bước 0)
- [ ] Thuê H100 + **CPU ≥16, RAM ≥64GB** (Bước 1)
- [ ] scp bundle lên + giải nén (Bước 2)
- [ ] venv + `pip install -r requirements.txt` (Bước 3)
- [ ] `data_loader.py` chạy OK (Bước 4)
- [ ] Thử cpu vs cuda (Bước 5)
- [ ] Đo runtime/memory đúng chuẩn (Bước 6)
- [ ] Kéo kết quả về (Bước 7)
- [ ] **Destroy instance** (Bước 8)
