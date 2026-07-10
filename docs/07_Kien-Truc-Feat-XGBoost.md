# 07 — Kiến trúc lõi Feat-XGBoost (bám đúng code gốc)

> Tài liệu này viết SAU KHI đọc trực tiếp code gốc `tuneEngAssistXGBAv51.py`
> (method = `bffeatxgbAv51`, chính là phiên bản dùng trong bài báo). Để em duyệt logic
> TRƯỚC khi code `src/models/feat_xgboost.py`.

## ⚠️ HAI ĐIỀU PHẢI SỬA so với hiểu lầm trước đây

**1. FE KHÔNG áp cả cụm mỗi bước — mà XOAY VÒNG (round-robin) một phép mỗi bước.**
Code gốc: `_feat_type = names[i % 6]`. Ở bước boosting `i`, nó chỉ dùng **đúng 1 phép FE**
theo chu kỳ `['None','auto','hpca','robuster','randP','minmax']`. Tức baseline gốc đã là
"1 FE mỗi bước, xoay vòng cứng" — KHÔNG phải stack cả 6.
→ Điều này làm **hook cho bandit cực gọn**: bandit chỉ việc **thay `i % 6` bằng lựa chọn
có học**. Đây chính là novelty, không phải "bỏ Mix rồi thay cụm FE".

**2. FE được TÍNH TRƯỚC (precompute) cho TOÀN BỘ dataset, không fit trên tập mẫu khó.**
Code gốc load sẵn 6 phiên bản đã biến đổi của cả (train/test/val) từ `Featdata/`. Mỗi bước
chỉ INDEX phiên bản đã biến đổi theo chỉ số GOSS. → Nhẹ và nhanh. Ta sẽ làm y vậy: fit FE
trên train, transform train/val/test một lần cho mỗi phép, cache lại.

**3. (chỉnh nhỏ) GOSS trong CODE dùng a=0.4, b=0.3** (không phải 0.45/0.35 như text bài báo).
Để khớp KẾT QUẢ của họ, dùng giá trị code: **a=0.4, b=0.3**.

## Ánh xạ tên FE (code gốc → module của mình)

| Code gốc | Ý nghĩa | Class của mình |
|---|---|---|
| `None` | giữ nguyên | `identity` |
| `auto` | Autofeat (vòng đầu) | `autofeat` |
| `hpca` | HT-SVD / PCA hard-threshold | `ht_svd` |
| `robuster` | RobustScaler | `robust` |
| `randP` | Random Projection | `random_proj` |
| `minmax` | MinMaxScaler | `minmax` |

> Giữ đúng THỨ TỰ này khi làm baseline round-robin để tái hiện chính xác.

## Luồng dữ liệu (train)

```
X_train, y_train (nhãn 0..K-1)
   │
   ├─ precompute: với mỗi φ ∈ Φ (6 phép):
   │     φ.fit(X_train) ; Z_train[φ]=φ(X_train), Z_val[φ]=φ(X_val), Z_test[φ]=φ(X_test)
   │
   ▼
 Fm = 0  (ma trận logit N×K)               # margin tích luỹ cho toàn train
 M  = []                                    # tập mô hình: (weak_learner, feat_id)
 for i in range(n_boost=100):
     p   = softmax(Fm)                       # xác suất hiện tại
     g   = p - onehot(y)                     # gradient bậc 1 (N×K)
     idx, w = GOSS(||g|| theo hàng, a=0.4, b=0.3)   # bóc tập mẫu khó + phần dễ có trọng số
     ┌─────────────────────────────────────────────────────────┐
     │ feat_id = SELECT_FE(i, context)   ◄── ĐIỂM CẮM BANDIT     │
     │   • baseline round-robin : feat_id = i % 6               │
     │   • dynamic (novelty)    : feat_id = bandit.select(ctx)  │
     └─────────────────────────────────────────────────────────┘
     Z = Z_train[feat_id]                    # phiên bản đã biến đổi (precomputed)
     τ = XGBClassifier(n_estimators=1, objective=multi:softprob, num_class=K, **cfg)
     τ.fit(Z[idx], y[idx], sample_weight=w, base_margin=Fm[idx])   # 1 round, warm-start
     Fm += τ.predict(Z, output_margin=True)  # cập nhật margin cho TOÀN train
     M.append((τ, feat_id))
     # [dynamic] reward = mức giảm loss/tăng acc val → bandit.update(feat_id, reward, ctx)
 return M
```

## Luồng dữ liệu (predict)

```
Fm_test = 0
for (τ, feat_id) in M:
    Fm_test += τ.predict(Z_test[feat_id], output_margin=True)
y_pred = argmax(softmax(Fm_test))
```

Điểm mấu chốt (khớp Algorithm 2): **mỗi weak learner τ_l được ghép cố định với phép biến
đổi φ_{feat_id} của nó**; lúc test phải dùng lại ĐÚNG không gian đó (`Z_test[feat_id]`).

## Thiết kế module

```python
# src/models/feat_xgboost.py
class FeatXGBoost:
    def __init__(self, n_boost=100, a=0.4, b=0.3, fe_names=FE_ARMS,
                 selector=None, xgb_cfg=None, seed=0): ...
    def _precompute_fe(self, Xtr, ytr, Xval, Xte):
        # fit mỗi φ trên train, transform & cache 3 tập -> self.Z['train'/'val'/'test'][name]
    def fit(self, Xtr, ytr, Xval=None, Xte=None):
        # vòng boosting ở trên; gọi self.selector.choose(step, context) mỗi bước
    def predict_margin(self, which='test'): ...
    def predict(self, which='test'): ...

# Bộ chọn FE — TRỪU TƯỢNG HOÁ để cắm bandit không phá cấu trúc
class FESelector(ABC):
    def choose(self, step:int, context)->int: ...
    def reward(self, feat_id:int, reward:float, context): ...  # no-op cho baseline

class RoundRobinSelector(FESelector):   # baseline gốc: step % n
class FixedSelector(FESelector):        # ablation A: luôn 1 phép
class RandomSelector(FESelector):       # ablation B: random mỗi bước
# src/models/dynamic_feat.py
class BanditSelector(FESelector):       # NOVELTY: bọc LinUCB từ bandit.py
```

→ `FeatXGBoost` nhận `selector` qua tham số. **Baseline và Dynamic dùng CHUNG một lớp
`FeatXGBoost`, chỉ khác `selector`** — đảm bảo so sánh công bằng tuyệt đối & không phá code.

## Ablation (ĐÃ CHỈNH cho khớp code gốc)

| Cấu hình | selector | Ghi chú |
|---|---|---|
| A. Fixed-cheap | `FixedSelector('random_proj')` | luôn 1 FE rẻ |
| B. Random | `RandomSelector` | random mỗi bước |
| **C. Round-robin (BÀI GỐC)** | `RoundRobinSelector` | `i % 6` — đây MỚI là baseline gốc |
| **D. Dynamic (CỦA MÌNH)** | `BanditSelector(LinUCB)` | bandit chọn theo context |

> Sửa quan trọng: baseline gốc là **round-robin**, KHÔNG phải "stack cả 6 FE". Tài liệu 02/03
> cần cập nhật theo đây.

## Context cho bandit (rẻ để tính)

Tại bước `i`, tính từ tập mẫu khó `idx`:
`[N_hard/N, mean|g|, std|g|, n_features_gốc(chuẩn hoá), entropy phân bố lớp trong idx]`
→ vector ~5 chiều, rẻ. Đây là `context_dim` cho `LinUCB`.

## Reward cho bandit (câu hỏi nghiên cứu — thử đơn giản trước)

Sau bước `i`: `reward = (val_loss trước) − (val_loss sau)` (mức giảm loss trên validation),
hoặc mức tăng val-accuracy. Bắt đầu bằng bản đơn giản, kiểm tra bandit có hội tụ ổn định.

## Điểm cần em DUYỆT trước khi thầy code

1. Đồng ý dùng **precompute FE toàn tập** (giống gốc) thay vì fit trên tập mẫu khó? (Nhẹ hơn.)
2. Đồng ý GOSS **a=0.4, b=0.3** (theo code) thay vì 0.45/0.35 (theo text)?
3. Đồng ý tách `FESelector` làm hook — baseline & dynamic dùng chung `FeatXGBoost`?
4. Weak learner giữ đúng gốc: `XGBClassifier(n_estimators=1, multi:softprob)` mỗi bước?
```
