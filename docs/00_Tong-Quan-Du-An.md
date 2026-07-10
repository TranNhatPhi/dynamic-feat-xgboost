# 00 — Tổng quan dự án & bản đồ tài liệu

## Dự án là gì

Cải tiến bài báo **"Enhancing sampling performance in XGBoost by ensemble feature
engineering"** (Feat-XGBoost / Mix-XGBoost, *Pattern Recognition* 2026).

- **GVHD:** thầy Lý Quang Vinh
- **Mục tiêu:** giữ độ chính xác tương đương bài gốc, nhưng **giảm thời gian chạy &
  bộ nhớ**, đồng thời thêm **tính mới (novelty)**: chọn động (dynamic) kỹ thuật feature
  engineering trong từng bước boosting thay vì áp cả cụm.
- **Đích nhắm:** **Smart-FE Framework** — nhắm **Q1** (chấp nhận rủi ro), fallback Q2/Q3
  nếu kết quả không đủ mạnh. Định hướng đã được advisor chỉnh để tránh "tham mà nông".

## Tóm tắt bài báo gốc

| Hạng mục | Nội dung |
|---|---|
| Ý tưởng | Nhúng feature engineering (FE) vào **từng bước boosting** của XGBoost, không chỉ ở tiền xử lý |
| Lấy mẫu | GOSS (Gradient-based One-Side Sampling): giữ mẫu gradient lớn (khó) + phần nhỏ mẫu gradient nhỏ |
| Bộ FE | Autofeat, Random Projection, HT-SVD (PCA hard-threshold), RobustScaler, MinMaxScaler + Identity |
| Hai mô hình | **Feat-XGBoost** (lõi) và **Mix-XGBoost** (chạy song song Feat + XGBoost gốc, chọn cái tốt hơn) |
| Dữ liệu | 61 dataset UCI (từ bộ 121 split chuẩn) |
| So sánh | 12 baseline classifier |

## Đóng góp mới của mình (Smart-FE Framework)

**Trụ chính (làm SÂU):**
1. **Chọn động FE bằng contextual bandit** — mỗi bước boosting, một agent nhìn đặc trưng
   của tập mẫu khó và chọn *đúng 1* phép FE tối ưu, thay vì áp cả cụm 6 phép của bài gốc.
   Biến kiến trúc "hộp đen cồng kềnh" thành **tự thích nghi (self-adaptive)**.
2. **Bằng chứng hiệu năng:** giảm rõ rệt thời gian chạy & bộ nhớ so với Feat-XGBoost gốc,
   giữ accuracy cạnh tranh — đo trên **cùng phần cứng**, có **ablation** đầy đủ.
3. **Chính sách chọn FE diễn giải được (interpretable):** thống kê phép FE nào được chọn
   theo đặc tính dataset (nhiễu → HT-SVD, v.v.).

**Bonus (làm NHẸ, 1 thí nghiệm nhỏ — KHÔNG để đội chi phí):**
4. **Tính tổng quát:** chứng minh khung cắm được vào ≥1 base learner khác (vd LightGBM),
   dùng tín hiệu nội bộ (gradient/feature importance) — cho thấy hướng mở rộng, không làm full.

> ⚠️ **Guardrail đặt tên (advisor nhấn mạnh):** gọi là *contextual bandit* (KHÔNG phải "RL");
> gọi là *interpretable selection policy* (KHÔNG overclaim "XAI"). Sai tên = mất điểm với reviewer.
> Ba trụ dàn hàng ngang mỗi cái nông = rủi ro rớt Q1; trụ chính sâu + bonus nhẹ = chắc tay.

## Ràng buộc PHẢI nhớ

- Claim "nhanh hơn / ít RAM hơn" → **bắt buộc chạy lại Feat-XGBoost gốc trên cùng máy**,
  KHÔNG copy runtime từ bài báo.
- Accuracy có thể lấy từ bảng gốc **nếu** dùng đúng split đã public.
- Phải có **ablation**: fixed-cheap-FE vs random-FE vs dynamic-FE (chứng minh MAB đáng giá).

## Bản đồ tài liệu (đọc theo thứ tự)

| File | Nội dung |
|---|---|
| `00_Tong-Quan-Du-An.md` | (file này) bức tranh tổng thể + roadmap |
| `01_Quy-Trinh-Xu-Ly-Du-Lieu.md` | Lấy & xử lý data, định dạng, cách dùng loader |
| `02_Giai-Thich-Thuat-Toan.md` | Giải thích thuật toán (để hiểu & trình bày với thầy) |
| `03_Pipeline-Thuc-Nghiem.md` | Toàn bộ pipeline: train → đánh giá → benchmark → ablation |
| `04_Huong-Dan-Chay-H100.md` | Deploy & chạy trên Vast.ai H100 |
| `05_Ke-Hoach-Viet-Bai.md` | Kế hoạch thí nghiệm & viết paper Q4 |

## Trạng thái hiện tại (2026-07-10)

- [x] Lấy data 121 dataset UCI, giải nén, gộp phẳng, overlay bản corrected
- [x] Viết `src/data_loader.py` (numpy thuần), verify khớp meta gốc
- [x] Dựng `.venv` (numpy, sklearn, xgboost)
- [ ] Viết pipeline FE + train Feat-XGBoost
- [ ] Cơ chế chọn động FE (novelty)
- [ ] Benchmark runtime/memory + ablation
- [ ] Viết paper

## Lộ trình đề xuất

1. **Giai đoạn 1 (miễn phí, trên Mac):** viết & debug toàn bộ pipeline với 3-5 dataset nhỏ.
2. **Giai đoạn 2 (Mac, qua đêm):** chạy full baseline + Feat-XGBoost gốc để tái hiện.
3. **Giai đoạn 3 (cloud nếu cần):** chạy full 61 dataset + phương pháp mới, đo benchmark.
4. **Giai đoạn 4:** phân tích, vẽ bảng/biểu đồ, viết paper.
