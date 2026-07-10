# 08 — Báo cáo đổi hướng (để bàn với thầy Vinh)

> Tóm tắt trung thực kết quả thực nghiệm và đề xuất đổi luận điểm bài báo. Viết để trình bày
> với GVHD trước khi đầu tư viết paper / thuê cloud.

## 1. Ta đã định làm gì

Thay lịch xoay vòng cứng `i%6` chọn feature engineering của Feat-XGBoost bằng **contextual
bandit tự học** — kỳ vọng: giữ accuracy, giảm chi phí, có "chính sách chọn FE diễn giải được".

## 2. Ta đã tìm ra gì (bằng chứng)

**(a) Chi phí giảm là THẬT và phổ quát.** Bandit (và bất kỳ bộ chọn thiên về FE rẻ) giảm
~**60-63% chi phí FE** so với round-robin gốc, trên cả 15/15 tập HARD_SET. Không cãi được.

**(b) NHƯNG bandit KHÔNG có lý do tồn tại — phép thử sống-chết thất bại.**
So Bandit vs **Fixed-ht_svd** (luôn dùng một phép HT-SVD), 6 tập × 4 fold:
- Δ(Bandit − Fixed-ht_svd) = **−0.002 ± 0.015** (bandit thắng 3/6, thua 3/6).
- Fixed-ht_svd **RẺ HƠN** bandit ở mọi tập (không tốn warm-up + khám phá).
→ Chỉ cần "luôn dùng HT-SVD" là **bằng bandit về accuracy mà rẻ hơn**. Bộ máy bandit là **thừa**.

**(c) FE tối ưu CÓ biến thiên theo bước boosting (quan sát đẹp).** Phân tích oracle (thử cả
6 FE mỗi bước, chọn phép giảm loss nhất):
- hill-valley: early = **HT-SVD** (khử nhiễu) → late = **identity/autofeat** (giữ/mở rộng).
- Các tập khác cũng đổi phép theo giai đoạn.

**(d) NHƯNG "chọn thông minh" không thắng "chọn ngu".** Ngay cả oracle (greedy) còn:
- Thua Fixed-ht_svd trên libras (−0.056).
- Thua Round-robin (xoay vòng ngẫu nhiên) trên arrhythmia (0.779 vs 0.797).
→ Cái giúp ích là **sự ĐA DẠNG FE**, không phải sự **THÔNG MINH** của lựa chọn. Tín hiệu chọn
per-step quá nhiễu để học online có lợi.

## 3. Vì sao novelty ban đầu sập

Time-penalty làm **một phép rẻ-mà-đủ-tốt (HT-SVD) thắng ở mọi nơi** → không có sự biến thiên
trong *lựa chọn tối ưu toàn cục* để bandit khai thác. Adaptivity không có gì để thích nghi.

## 4. Luận điểm MỚI (trung thực) — bài nghiên cứu thực nghiệm

> **"Đánh giá lại feature engineering nhúng trong gradient boosting: cụm FE có thực sự cần
> chọn động?"**

Ba đóng góp (đều là sự thật, kiểm chứng được):
1. **Cụm 6-FE round-robin của Feat-XGBoost phần lớn là thừa** — một phép phổ đơn (HT-SVD)
   nhúng trong boosting đạt accuracy tương đương trên đa số tập với ~⅓ chi phí tính toán.
2. **FE tối ưu biến thiên theo bước boosting** (khử nhiễu sớm → tinh chỉnh muộn) — chứng minh
   qua phân tích oracle. Đây là hiểu biết mới về *động học* của FE trong boosting.
3. **Học chọn FE online (contextual bandit) KHÔNG mang lại lợi thế ổn định** so với baseline
   đơn giản, vì tín hiệu per-step quá nhiễu. Ta phân tích *khi nào* đa dạng FE giúp ích và
   *khi nào* không.

## 5. Định vị & đích công bố (điều chỉnh trung thực)

- Đây là **empirical study + negative/simplification result** — loại bài *có giá trị* trong
  cộng đồng (chỉ ra phương pháp phức tạp là không cần thiết, tiết kiệm chi phí).
- **Thực tế: Q3/Q4** (không phải Q1). Q1 đòi novelty phương pháp mạnh mà ta không có.
- Điểm mạnh để bán: **tính trung thực + rigor** (paired t-test, oracle analysis, ablation
  đầy đủ), và **giá trị thực tiễn** (giảm ⅔ chi phí mà giữ accuracy).

## 6. Việc cần làm để hoàn thành bài (Direction B)

| # | Việc | Chứng minh cho claim |
|---|---|---|
| 1 | So Round-robin / Fixed-ht_svd / Bandit trên TẤT CẢ dataset × 4 fold × 3 seed, mean±std | Claim 1 + 3 |
| 2 | Phân tích oracle trên nhiều dataset, thống kê dịch chuyển FE early→late | Claim 2 |
| 3 | Benchmark runtime/RAM thật (không chỉ cost proxy) trên tập con | Claim 1 (chi phí) |
| 4 | Optuna cho các cấu hình chính (để số tuyệt đối đáng tin) | tất cả |
| 5 | Viết: Method (mô tả 3 baseline) + Study + Discussion (khi nào đa dạng giúp) | — |

## 7. Câu hỏi mở cho thầy Vinh

- Thầy chấp nhận hạ đích xuống Q3/Q4 với một bài trung thực, hay muốn thử thêm hướng thiết kế
  lại cơ chế (đổi cost proxy/context) để cố cứu novelty (rủi ro cao, có thể tốn nhiều vòng)?
- Có muốn giữ "bandit" như một *baseline bị bác bỏ* trong bài (để câu chuyện đầy đủ) không?
