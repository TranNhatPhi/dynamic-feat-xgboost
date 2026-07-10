# 09 — Thuyết minh đề tài (Hướng B — ĐÃ CHỐT)

## Tên đề tài

- **Tiếng Việt:** *Đánh giá lại feature engineering nhúng trong gradient boosting:
  cụm biến đổi đặc trưng có thực sự cần chọn động?*
- **English:** *Revisiting Embedded Feature Engineering in Gradient Boosting:
  Is Adaptive Selection Necessary? — An Empirical Study*

## Loại bài & đích

Bài **nghiên cứu thực nghiệm & phản biện** (empirical study / negative-result +
practical recommendation). Đích thực tế: **Q3/Q4**.

## Tính cấp thiết

Feat-XGBoost (*Pattern Recognition* 2026) nhúng một **cụm 6 phép FE** (xoay vòng) vào từng
bước boosting để tăng accuracy — nhưng **chi phí tính toán lớn** và **chưa ai kiểm chứng
từng thành phần có thực sự cần thiết**. Bài này trả lời: *cần bao nhiêu FE, có cần chọn
động không, và vì sao?*

## Câu hỏi nghiên cứu

- RQ1: Đa dạng FE trong boosting có thực sự giúp (so với 1 FE và so với XGBoost thường)?
- RQ2: Có cần cả cụm 6 FE, hay một tập con rẻ là đủ?
- RQ3: Chọn FE **động/học được** (contextual bandit) có hơn chọn **cố định/ngẫu nhiên** không?
- RQ4: Phép FE tối ưu có biến thiên theo bước boosting không? Nếu có, khai thác được không?

## Phương pháp

Một khung `FeatXGBoost` chung, thay **bộ chọn FE** (FESelector) để so công bằng 6 cấu hình
trên **15 tập UCI nhiều chiều × 4 fold × 3 seed**, báo cáo **mean±std + paired t-test**:
plain-XGBoost · round-robin (gốc) · fixed-1-FE · cheap-rotation (bỏ autofeat) ·
2-FE rotation · contextual bandit. Kèm **phân tích oracle** (RQ4) và **benchmark chi phí**.

## Kết quả chính (đã có)

1. **RQ1 — Đa dạng FE giúp:** round-robin > plain-XGBoost (Δ+0.024, **t=+2.57**, 12/15);
   và > fixed-1-FE (thắng 11/15). ✔
2. **RQ2 — Không cần cả cụm:** bỏ autofeat (cheap-rotation) giữ accuracy ~ngang round-robin
   mà **giảm RAM thật ~57%** (benchmark: 5.06MB→2.19MB) và thời gian nhẹ hơn. ✔
3. **RQ3 — Chọn động KHÔNG đáng, thậm chí TỆ HƠN:** bandit **huề** accuracy với cheap-rotation
   (Δ−0.001, **t=−0.37**, 9/0/6). Benchmark THẬT còn cho thấy bandit **KHÔNG rẻ hơn** — nó phải
   precompute CẢ 6 FE (gồm autofeat) làm "thực đơn" → **RAM = round-robin (5.06MB), gấp ~2.3× cheap-rotation**,
   thời gian cũng hơi hơn. cheap-rotation **thống trị hoàn toàn** bandit. (Cost-proxy cũ 55 là artifact,
   chỉ đếm arm được chọn.) ✔
4. **RQ4 — Cấu trúc thời gian YẾU (củng cố RQ3):** oracle gộp 12 tập cho thấy phân bố FE
   theo giai đoạn **gần như phẳng** (mỗi phép 13–36%, không phép nào áp đảo). Chỉ có một
   xu hướng đơn điệu nhẹ: autofeat giảm dần theo bước (22%→18%→13%). Quy luật "khử nhiễu
   sớm → giữ nguyên muộn" **chỉ đúng cho hill-valley, KHÔNG tổng quát.** → Cấu trúc thời gian
   yếu + tín hiệu per-step nhiễu **giải thích vì sao chọn động không có lợi** (RQ3). ✔

## Đóng góp

- **Bóc trần** phần lãng phí của Feat-XGBoost (autofeat) → khuyến nghị thực tiễn: dùng
  rotation FE rẻ, giảm ~44% chi phí, giữ accuracy.
- **Bằng chứng thống kê** rằng chọn FE học-online không hơn baseline đơn giản trong bối
  cảnh này, kèm **giải thích nguyên nhân** (nhiễu tín hiệu per-step) qua oracle analysis.
- Một **khung so sánh công bằng** (cùng lõi, chỉ khác bộ chọn) tái lập được.

## Threats to validity / Limitations (nêu chủ động)

- Chi phí hiện đo bằng proxy (số chiều đầu ra) → sẽ bổ sung **runtime & peak-RAM thật**.
- Giới hạn ở 15 tập nhiều chiều + XGBoost; chưa mở rộng base learner khác.
- Chưa Optuna cho số tuyệt đối → sẽ tune cho cấu hình chính đưa vào bảng.

## Việc còn lại

1. Đinh cuối 2-FE (đang chạy) → hoàn tất bằng chứng RQ3.
2. Optuna cho cấu hình chính → số accuracy/F1 cho Bảng 1.
3. Benchmark runtime/RAM thật → Bảng 2.
4. Vẽ hình oracle-timeline (RQ4) + heatmap.
5. Viết theo cấu trúc [05_Ke-Hoach-Viet-Bai.md](05_Ke-Hoach-Viet-Bai.md) (điều chỉnh cho study).
