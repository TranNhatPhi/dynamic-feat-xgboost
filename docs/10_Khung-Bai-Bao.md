# 10 — Khung bài báo (Empirical Study) — nháp, đắp thịt sau khi thầy Vinh duyệt

> Số liệu trong khung này dùng **benchmark THẬT** (không dùng cost-proxy cũ).
> Headline chi phí: **RAM −57%** (cheap 2.19MB vs round 5.06MB), time −15%.

## 1. Abstract
- **Vấn đề:** Feat-XGBoost nhúng cụm 6 FE (xoay vòng) vào boosting → tăng accuracy nhưng lãng phí tính toán.
- **Phương pháp:** Nghiên cứu thực nghiệm 15 tập UCI khó × 4 fold × 3 seed; 6 cấu hình đối chứng cùng lõi.
- **Kết quả:** (1) đa dạng FE giúp (round > plain, t=+2.57); (2) **bỏ autofeat → RAM −57%, accuracy không đổi**
  (Δ=−0.16đ, t=−0.56); (3) chọn động online (bandit) **không lợi**, còn tốn RAM ngang round-robin.
- **Khuyến nghị:** dùng rotation tĩnh các FE rẻ (cheap-rotation).

## 2. Introduction
- Vai trò FE trong gradient boosting; cách Feat-XGBoost nhúng FE vào boosting.
- **Gap:** chưa ai kiểm chứng từng thành phần FE có cần thiết, và chọn động có đáng không.
- Tuyên bố **RQ1–RQ4** (xem docs/09).

## 3. Materials and Methods
- 6 cấu hình: plain(0) / fixed-HT-SVD(1) / 2-FE / cheap(5) / round-robin(6, gốc) / bandit(adaptive).
  Cùng lõi `FeatXGBoost`, chỉ khác bộ chọn FE (đảm bảo công bằng).
- Data: 15 tập HARD_SET UCI (nhiều chiều), split chuẩn, loại low-res-spect.
- Giao thức: 4-fold CV × 3 seed; weak learner = 1 vòng XGBoost multi:softprob; GOSS a=0.4/b=0.3.
- Chỉ số: Accuracy + F1; Runtime thật (perf_counter) + Peak/precompute RAM — cùng một máy.
- Nêu rõ: chưa Optuna (dùng hyperparameter cố định) → so sánh TƯƠNG ĐỐI, đủ cho câu hỏi nghiên cứu.

## 4. Results (trái tim)
- **RQ1 & RQ2 — Bảng 1 (accuracy) + Bảng 2 (RAM/time):** đa dạng FE cần thiết; nhưng cụm 6-FE lãng phí —
  bỏ autofeat (cheap) giữ accuracy (t=−0.56) mà RAM −57%. Kèm t-test.
- **RQ3 — Hình 1 (Pareto):** cheap-rotation là "knee"; bandit **bị thống trị** (RAM ngang round-robin,
  accuracy thấp hơn cheap; t=−0.37 so cheap). Cost-proxy đánh lừa → benchmark thật mới đúng.

## 5. Discussion & Recommendation
- **RQ4 — vì sao bandit thất bại (điểm học thuật nhất):** phân tích oracle timeline cho thấy phân bố FE
  tối ưu **gần như phẳng** qua các bước (13–36%), chỉ autofeat giảm nhẹ 22→13%. Không có cấu trúc thời gian
  mạnh + tín hiệu per-step nhiễu → học online vô ích.
- **Khuyến nghị thực tiễn:** cộng đồng nên dùng rotation tĩnh các FE rẻ (bỏ autofeat), khỏi chọn động.
- **Limitations:** chưa Optuna (số tuyệt đối); giới hạn 15 tập nhiều chiều + XGBoost; time-saving đo trên
  data nhỏ nên khiêm tốn. Hướng mở rộng: base learner khác, dataset lớn hơn, đo trên máy nhiều-nhân.

## Tài sản đã có (điền vào bài)
- Bảng 1: `results/table1_accuracy.csv` · Bảng 2: `results/benchmark.csv`
- Hình 1: `results/fig1_pareto.png` · Dữ liệu oracle: `results/oracle_phase_summary.csv`
- Đề tài & RQ: docs/09 · Kế hoạch viết chi tiết: docs/05
