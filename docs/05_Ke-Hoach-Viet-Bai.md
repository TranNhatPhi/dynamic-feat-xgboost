# 05 — Kế hoạch thí nghiệm & viết bài (Q4)

## Định vị bài báo — Smart-FE Framework

- **Loại:** framework tự thích nghi cho feature engineering trong boosting (trên nền Feat-XGBoost).
- **Thông điệp chính (one-liner):** *"Một contextual bandit chọn động kỹ thuật feature
  engineering trong từng bước boosting biến kiến trúc FE-ensemble cồng kềnh thành khung
  tự thích nghi: giữ độ chính xác nhưng giảm rõ rệt thời gian & bộ nhớ, và cho ra một
  chính sách chọn FE diễn giải được."*
- **Đích:** **Q1** (chấp nhận rủi ro). Fallback Q2/Q3 nếu kết quả chưa đủ mạnh.

### ⚠️ Ba guardrail để KHÔNG bị đánh rớt Q1 (advisor)

1. **Đặt tên đúng:** *contextual bandit*, KHÔNG gọi "RL" (thiếu state-transition/credit
   assignment dài hạn → reviewer bắt lỗi ngay).
2. **Không overclaim XAI:** gọi là *interpretable selection policy* (thống kê chính sách),
   không phải XAI kiểu SHAP/LIME.
3. **Trụ chính sâu, bonus nhẹ:** làm thật sâu phần dynamic-FE trên XGBoost; phần "universal"
   chỉ 1 thí nghiệm nhỏ trên LightGBM. Ba trụ dàn ngang mỗi cái nông = rớt; đội chi phí train.

## Contribution (3 gạch đầu dòng trong Introduction)

1. **Contextual-bandit dynamic FE** nhúng trong boosting: mỗi bước chọn 1 phép FE tối ưu
   theo đặc trưng tập mẫu khó, thay cho việc áp cả cụm của bài gốc (self-adaptive).
2. **Giảm chi phí (thời gian + bộ nhớ)** giữ accuracy cạnh tranh — benchmark cùng phần cứng
   + ablation chứng minh bandit đáng giá.
3. **Chính sách chọn FE diễn giải được** + chứng minh **tính tổng quát** (mở rộng sang
   ≥1 base learner khác, làm nhẹ).

## Cấu trúc paper (chuẩn CS/AI)

| Mục | Nội dung cần có |
|---|---|
| Abstract | vấn đề → ý tưởng → kết quả chính (1 con số nổi bật) |
| Introduction | động cơ, gap, 3 contribution, câu chốt kết quả |
| Related Work | (a) boosting/XGBoost, (b) feature engineering tự động, (c) MAB trong ML — nhóm theo hướng, không liệt kê rời |
| Method | định nghĩa bài toán, GOSS, tập Φ, thuật toán MAB (pseudo-code), độ phức tạp |
| Experiments | dataset, baseline, metric, giao thức (Optuna, 4-fold), cấu hình máy |
| Results & Analysis | bảng accuracy, bảng runtime/memory, **ablation A/B/C/D**, biểu đồ |
| Discussion/Explainability | phép FE nào hay được chọn với data nào |
| Limitations | nêu chủ động (xem dưới) |
| Conclusion | tóm tắt + hướng phát triển |

## Bảng/biểu đồ tối thiểu cần có

1. **Bảng 1** — Accuracy (+F1) của XGB / Feat-XGBoost / Dynamic-FE trên các dataset.
2. **Bảng 2** — Runtime & peak memory (cùng máy), % giảm so với Feat-XGBoost gốc.
3. **Bảng 3** — Ablation A/B/C/D.
4. **Hình 1** — Sơ đồ kiến trúc (MAB chọn FE trong boosting).
5. **Hình 2** — Heatmap/bar: tần suất chọn mỗi phép FE theo nhóm dataset (interpretable policy).
6. **Bảng 4** — Generality: áp khung lên LightGBM (accuracy + runtime) trên vài dataset — bonus.
7. (Tùy) **Hình 3** — đường cong accuracy vs thời gian.

## Threats to validity (phải tự nêu trong Limitations)

- **Đo hiệu năng:** phụ thuộc phần cứng → nêu rõ cấu hình, chạy nhiều lần, báo mean±std.
- **Overhead của MAB:** phải chứng minh tổng runtime vẫn giảm dù thêm agent.
- **Chọn dataset:** nếu không chạy đủ 61, phải nói rõ tiêu chí chọn (vd chọn cái lớn/khó).
- **Số seed:** kết quả ổn định qua ≥3-5 seed.
- **Autofeat:** kết quả có thể phụ thuộc phiên bản thư viện → khoá version.

## Chiến lược chi phí (đã bàn với thầy)

- Debug pipeline trên Mac (miễn phí) TRƯỚC khi lên cloud.
- Accuracy baseline: có thể trích từ bảng gốc nếu dùng đúng split.
- Runtime/memory: bắt buộc tự chạy Feat-XGBoost gốc + phương pháp mới trên **cùng 1 máy**.
- Nếu chọn không chạy đủ 61 dataset → chọn ~15 cái lớn/nhiều chiều, lập luận rõ mục tiêu
  là chứng minh tối ưu chi phí (reviewer chấp nhận nếu trình bày minh bạch).

## Mốc thời gian gợi ý (điều chỉnh theo em)

| Tuần | Việc |
|---|---|
| 1 | Code xong FE + GOSS + Feat-XGBoost tái hiện, chạy trên Mac (dataset nhỏ) |
| 2 | Kiểm chứng accuracy tái hiện ≈ bài gốc; code MAB dynamic |
| 3 | Ablation A/B/C/D trên tập nhỏ; hoàn thiện benchmark |
| 4 | Chạy full (cloud nếu cần) + thu kết quả |
| 5-6 | Phân tích, vẽ hình, viết bản thảo |
| 7 | Rà soát, thầy góp ý, chỉnh sửa, nộp |

## Nhắc nhở học thuật

- Không bịa số liệu / tên bài. Related work diễn giải bằng lời mình, trích tối đa 1 câu ngắn có nguồn.
- Tránh lạm dụng "novel", "state-of-the-art" khi chưa có bằng chứng.
- Kiểm tra lại số liệu/năm xuất bản trước khi đưa vào bản thảo.
