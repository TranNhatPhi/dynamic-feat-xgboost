# Tài liệu dự án Dynamic Feat-XGBoost (Smart-FE Framework)

Đọc theo thứ tự:

1. [00 — Tổng quan dự án & bản đồ tài liệu](00_Tong-Quan-Du-An.md)
2. [01 — Quy trình xử lý dữ liệu](01_Quy-Trinh-Xu-Ly-Du-Lieu.md)
3. [02 — Giải thích thuật toán](02_Giai-Thich-Thuat-Toan.md)
4. [03 — Pipeline thực nghiệm](03_Pipeline-Thuc-Nghiem.md)
5. [04 — Hướng dẫn chạy trên H100](04_Huong-Dan-Chay-H100.md)
6. [05 — Kế hoạch thí nghiệm & viết bài](05_Ke-Hoach-Viet-Bai.md)

## Định hướng chốt

**Smart-FE Framework** — nhắm Q1 (rủi ro), fallback Q2/Q3:
- Trụ chính (sâu): contextual bandit chọn động FE trong boosting + benchmark runtime/memory + ablation + interpretable policy.
- Bonus (nhẹ): chứng minh tính tổng quát trên 1 base learner khác (LightGBM).

**3 guardrail:** (1) gọi *contextual bandit* không phải "RL"; (2) *interpretable policy*
không overclaim "XAI"; (3) trụ chính sâu, bonus nhẹ — tránh tham mà nông + đội chi phí train.
