# 02 — Giải thích thuật toán (để hiểu & trình bày với thầy)

## 1. Nền tảng: XGBoost & Boosting

XGBoost xây dựng **một chuỗi cây quyết định nối tiếp** (gradient boosting). Cây thứ `t`
học để sửa **phần dư (residual/gradient)** mà `t-1` cây trước còn sai. Về bản chất là
một vòng lặp: mỗi vòng thêm một cây nhắm vào chỗ mô hình đang yếu.

- Độ phức tạp thời gian gốc: `O(K · d · ||x||₀ · log n)`; với block structure + presorting
  giảm còn `O(K · d · ||x||₀)`.
- Nhược điểm về FE: bình thường feature engineering chỉ làm **một lần ở tiền xử lý**,
  không can thiệp vào quá trình dựng cây → bỏ lỡ tiềm năng trên các mẫu khó.

## 2. GOSS — Gradient-based One-Side Sampling

Thay vì dùng toàn bộ data ở mỗi bước, GOSS:

- Giữ lại **phần lớn mẫu có gradient lớn** (mẫu "khó", đang bị dự đoán sai) — bài gốc dùng
  `a = 0.45` (45%).
- Lấy **ngẫu nhiên một phần nhỏ mẫu gradient nhỏ** (mẫu "dễ") — `b = 0.35` (35%), và
  **tăng trọng số** cho nhóm này để **bảo toàn phân phối gốc** của dữ liệu.

→ Giảm kích thước data mỗi vòng (nhẹ RAM + nhanh) mà không lệch phân phối.

## 3. Bộ feature engineering (tập Φ)

Bài gốc dùng 5 phép biến đổi + 1 phép giữ nguyên:

| Φᵢ | Kỹ thuật | Vai trò |
|---|---|---|
| φ₀ | Identity | giữ nguyên data thô |
| φ₁ | **Autofeat** | tự sinh đặc trưng phi tuyến; giới hạn ở vòng đầu, 4 phép: `1/x, x², x³, exp(x)` |
| φ₂ | **Random Projection** | giảm chiều bằng ma trận Gaussian ngẫu nhiên; bảo toàn khoảng cách (Johnson–Lindenstrauss) |
| φ₃ | **HT-SVD** | PCA/SVD + hard-threshold, cắt giá trị kỳ dị nhỏ để bỏ nhiễu trắng |
| φ₄ | **RobustScaler** | chuẩn hoá theo median + IQR, mạnh với outlier |
| φ₅ | **MinMaxScaler** | co giãn tuyến tính về [0,1] |

## 4. Feat-XGBoost — nhúng FE vào từng bước boosting

Vòng lặp (Algorithm 2 của bài gốc), diễn giải:

```
for mỗi bước boosting t:
    g ← gradient hiện tại
    X' ← GOSS(X, g)                # bóc tập con nhỏ, tập trung mẫu khó
    X'_fe ← Φ(X')                  # áp (các) phép feature engineering
    dựng cây tₜ trên X'_fe         # cây học trên không gian dễ phân tách hơn
    cập nhật mô hình
    # (giải phóng bộ đệm X', X'_fe trước vòng sau)
```

Ý nghĩa: mỗi vòng "mở khoá" các điểm dữ liệu khó bằng cách kéo chúng sang không gian mới
dễ phân tách, rồi mới dựng cây → mô hình sửa lỗi hiệu quả hơn.

## 5. Mix-XGBoost

Chạy song song **XGBoost gốc** và **Feat-XGBoost**, so trên validation, chọn cái tốt hơn
để dự đoán. Độ chính xác cao nhất **nhưng tốn gấp đôi thời gian** → mình sẽ **bỏ** để đạt
mục tiêu nhanh/nhẹ.

## 6. ĐIỂM MỚI của mình: chọn động FE bằng contextual bandit (Smart-FE)

> **Đặt tên đúng (quan trọng):** đây là **contextual bandit**, KHÔNG phải "RL" đầy đủ
> (không có chuyển trạng thái / credit assignment dài hạn). Gọi sai sẽ bị reviewer bắt lỗi.

**Vấn đề của bài gốc:** áp cả cụm 6 phép Φ ở mỗi bước → tốn tính toán, và họ tự thừa nhận
"khó đánh giá đóng góp riêng của từng phép FE".

**Ý tưởng:** đặt một **agent contextual bandit** trước mỗi bước dựng cây. Mỗi phép
FE là một "cánh tay" (arm). "Context" = đặc trưng của tập mẫu khó (số chiều, độ nhiễu,
mất cân bằng lớp...). Agent:

1. Nhìn **context** của tập khó `X'` hiện tại.
2. Chọn **1 phép FE** (cân bằng explore/exploit — ví dụ UCB1 hoặc ε-greedy).
3. Nhận "phần thưởng" = mức cải thiện (giảm loss / tăng accuracy trên validation, có trừ chi phí).
4. Cập nhật ước lượng giá trị của arm đó.

**Lợi ích → bán được cho reviewer:**
- **Nhanh & nhẹ:** mỗi bước chỉ chạy 1 phép FE thay vì 6.
- **Explainable:** log lịch sử arm được chọn → biết data nhiễu thường chọn HT-SVD, v.v.
- **Tự thích nghi:** không cần cấu hình cứng cho từng dataset.

**Ablation bắt buộc để chứng minh MAB đáng giá:**
| Cấu hình | Mô tả |
|---|---|
| A. Fixed-cheap-FE | luôn dùng 1 phép rẻ nhất (Random Projection) |
| B. Random-FE | mỗi bước chọn ngẫu nhiên 1 phép |
| C. All-FE (bài gốc) | áp cả cụm |
| D. **Dynamic-FE (của mình)** | MAB chọn động |

Nếu D không hơn A/B rõ rệt → MAB chỉ là phức tạp hoá. Đây là rủi ro novelty lớn nhất.

## 7. Câu hỏi phản biện phải chuẩn bị trước

- MAB có tự nó thêm overhead không? Tổng runtime có thực sự giảm?
- "Phần thưởng" định nghĩa thế nào cho không bị nhiễu giữa các bước boosting?
- Kết quả có ổn định qua nhiều seed không? (báo cáo mean ± std)
