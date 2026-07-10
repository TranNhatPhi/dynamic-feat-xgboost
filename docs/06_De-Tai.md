# 06 — Thuyết minh đề tài

> ❌ **ĐỀ TÀI NÀY ĐÃ BỊ THAY THẾ (SUPERSEDED).** Thực nghiệm 15 tập chứng minh contextual
> bandit không hơn baseline đơn giản (cheap-rotation) → đã chốt **Hướng B (empirical study)**.
> **Đề tài chính thức mới: [09_De-Tai-Huong-B.md](09_De-Tai-Huong-B.md).** Bản dưới giữ để lưu vết.

---

## Tên đề tài

- **Tiếng Việt:** *Chọn thích nghi kỹ thuật đặc trưng trong gradient boosting bằng
  contextual bandit*
- **English:** *Adaptive Feature-Engineering Selection in Gradient Boosting via
  Contextual Bandits*

## Tính cấp thiết / Vấn đề

Feature engineering (FE) thường quyết định hiệu quả mô hình học máy, nhưng theo cách
truyền thống chỉ làm ở tiền xử lý — không can thiệp vào quá trình học. Bài báo nền
(Feat-XGBoost, *Pattern Recognition* 2026) đã nhúng cả một **cụm FE** vào từng bước
boosting và cho kết quả tốt, nhưng bộc lộ 2 điểm yếu do chính tác giả thừa nhận:
(1) áp cả cụm FE mỗi bước → **tốn thời gian & bộ nhớ**; (2) **khó biết phép FE nào thực
sự đóng góp**. Đề tài giải quyết trực tiếp hai điểm yếu này.

## Mục tiêu

Thay việc áp cả cụm FE bằng một **agent contextual bandit** chọn *đúng một* phép FE tối
ưu ở mỗi bước boosting, dựa trên đặc trưng của tập mẫu khó hiện tại — nhằm **giữ độ chính
xác tương đương** bài gốc nhưng **giảm rõ rệt thời gian chạy và bộ nhớ**, đồng thời cho ra
một **chính sách chọn FE diễn giải được**.

## Đối tượng & phạm vi

- Đối tượng: gradient boosting (cụ thể XGBoost) + tập FE {Identity, Random Projection,
  HT-SVD, RobustScaler, MinMaxScaler, Autofeat}.
- Phạm vi: phân loại (classification) trên bộ benchmark UCI (61 dataset, split chuẩn).
  KHÔNG mở rộng model-agnostic ở giai đoạn này (chỉ 1 thí nghiệm nhỏ minh hoạ tính tổng quát).

## Phương pháp

1. Tái hiện Feat-XGBoost gốc làm mốc so sánh.
2. Thiết kế **contextual bandit (LinUCB)**: context = đặc trưng tập mẫu khó (số chiều,
   độ nhiễu, mất cân bằng...); arm = phép FE; reward = mức cải thiện trên validation.
3. Nhúng bandit vào vòng boosting (thay bước "áp cả cụm FE").
4. **Ablation** so 4 cấu hình: all-FE (gốc) / fixed-cheap / random-select / bandit.
5. **Benchmark** thời gian + bộ nhớ trên cùng phần cứng; phân tích  chính sách chọn FE.

## Nội dung công việc

- [x] Chuẩn bị & kiểm định dữ liệu (data_loader, preprocessing — đã xong)
- [x] Agent contextual bandit (đã xong, self-test)
- [ ] Cài đặt 6 phép FE (`src/fe/`)
- [ ] GOSS + Feat-XGBoost tái hiện
- [ ] Nhúng bandit → Dynamic Feat-XGBoost
- [ ] Chạy thí nghiệm + ablation + benchmark
- [ ] Viết báo cáo/paper

## Kết quả dự kiến & đóng góp

1. Cơ chế **contextual-bandit chọn động FE** trong boosting (tự thích nghi, thay cụm FE cứng).
2. **Giảm chi phí tính toán** (thời gian + bộ nhớ) mà giữ accuracy — chứng minh bằng benchmark.
3. **Chính sách chọn FE diễn giải được** (phép FE nào hợp loại dữ liệu nào).
4. (Bonus) Minh hoạ khả năng mở rộng sang base learner khác.

## Điểm mới so với bài gốc

| | Bài gốc (Feat-XGBoost) | Đề tài này |
|---|---|---|
| Chọn FE | áp cả cụm 6 phép mỗi bước | contextual bandit chọn 1 phép |
| Chi phí | cao (Mix-XGBoost gấp đôi) | giảm rõ rệt |
| Diễn giải | "khó đánh giá từng FE" | chính sách chọn FE tường minh |
| Thích nghi | cấu hình cứng | tự thích nghi theo dữ liệu |

## Đích công bố

Q1 (chấp nhận rủi ro), fallback Q2/Q3. Xem [05_Ke-Hoach-Viet-Bai.md](05_Ke-Hoach-Viet-Bai.md).
