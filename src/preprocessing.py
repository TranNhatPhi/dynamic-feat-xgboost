"""
Tiền xử lý tối thiểu + kiểm tra chất lượng cho từng Split.

Nguyên tắc (giữ đúng bài gốc): KHÔNG scale ở đây (scale nằm trong feature engineering
lúc train). Bước này chỉ đảm bảo data đủ điều kiện đưa vào XGBoost:

  1. Nhãn liên tục 0..K-1 (XGBoost bắt buộc). Remap nếu cần.
  2. X là float, không NaN/inf.
  3. Phát hiện lỗi split: class có trong val/test nhưng THIẾU trong train
     (đúng loại bug mà bài báo cảnh báo).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.data_loader import Split, UCIDataset, list_datasets


@dataclass
class QualityReport:
    dataset: str
    cv: int
    n_classes: int
    labels_contiguous: bool          # nhãn gốc đã là 0..K-1 chưa
    has_nan: bool
    has_inf: bool
    missing_class_in_train: list[int]  # class có ở val/test mà train không có
    ok: bool

    def summary(self) -> str:
        flags = []
        if not self.labels_contiguous:
            flags.append("nhãn-remap")
        if self.has_nan:
            flags.append("NaN")
        if self.has_inf:
            flags.append("Inf")
        if self.missing_class_in_train:
            flags.append(f"thiếu-class-train={self.missing_class_in_train}")
        return "OK" if not flags else ", ".join(flags)


def _label_map(*ys: np.ndarray) -> dict:
    """Bảng ánh xạ nhãn gốc -> 0..K-1, dựa trên union của mọi tập trong split."""
    all_labels = np.unique(np.concatenate([np.unique(y) for y in ys]))
    return {int(orig): i for i, orig in enumerate(all_labels)}


def prepare(split: Split, dtype=np.float32) -> Split:
    """Trả về Split đã sẵn sàng cho XGBoost: nhãn 0..K-1, X float sạch."""
    mapping = _label_map(split.y_train, split.y_val, split.y_test)

    def remap(y):
        return np.array([mapping[int(v)] for v in y], dtype=np.int64)

    return Split(
        X_train=np.ascontiguousarray(split.X_train, dtype=dtype),
        y_train=remap(split.y_train),
        X_val=np.ascontiguousarray(split.X_val, dtype=dtype),
        y_val=remap(split.y_val),
        X_test=np.ascontiguousarray(split.X_test, dtype=dtype),
        y_test=remap(split.y_test),
    )


def check(split: Split, name: str = "", cv: int = 0) -> QualityReport:
    """Kiểm tra chất lượng một split (dùng nhãn GỐC, trước khi remap)."""
    all_labels = np.unique(
        np.concatenate([split.y_train, split.y_val, split.y_test])
    ).astype(np.int64)
    contiguous = np.array_equal(all_labels, np.arange(all_labels.size))

    train_labels = set(np.unique(split.y_train).astype(int).tolist())
    eval_labels = set(np.unique(split.y_val).astype(int).tolist()) | set(
        np.unique(split.y_test).astype(int).tolist()
    )
    missing = sorted(eval_labels - train_labels)

    X_all = np.concatenate([split.X_train, split.X_val, split.X_test])
    has_nan = bool(np.isnan(X_all).any())
    has_inf = bool(np.isinf(X_all).any())

    return QualityReport(
        dataset=name,
        cv=cv,
        n_classes=all_labels.size,
        labels_contiguous=contiguous,
        has_nan=has_nan,
        has_inf=has_inf,
        missing_class_in_train=missing,
        ok=(contiguous and not has_nan and not has_inf and not missing),
    )


def load_prepared(name: str, cv: int = 0) -> tuple[Split, QualityReport]:
    """Tiện ích: nạp + kiểm tra + chuẩn bị trong một lần."""
    raw = UCIDataset(name).get_split(cv)
    report = check(raw, name, cv)
    return prepare(raw), report


def scan_all(datasets: list[str] | None = None) -> list[QualityReport]:
    """Quét mọi dataset × 4 fold, gom các báo cáo có cờ cần chú ý."""
    names = datasets or list_datasets()
    reports = []
    for name in names:
        try:
            ds = UCIDataset(name)
            for cv in range(ds.n_folds):
                reports.append(check(ds.get_split(cv), name, cv))
        except Exception as e:  # dataset đọc lỗi -> ghi nhận
            reports.append(
                QualityReport(name, -1, -1, False, False, False, [], ok=False)
            )
            print(f"  [LỖI ĐỌC] {name}: {repr(e)[:100]}")
    return reports


if __name__ == "__main__":
    print("Quét chất lượng toàn bộ dataset (121 × 4 fold)...\n")
    reports = scan_all()
    total = len(reports)
    ok = sum(r.ok for r in reports)
    print(f"Tổng {total} (dataset×fold): {ok} OK, {total - ok} cần chú ý\n")

    # gom theo dataset các cờ đáng chú ý
    flagged = {}
    for r in reports:
        if not r.ok:
            flagged.setdefault(r.dataset, set()).add(r.summary())
    if flagged:
        print("Dataset cần chú ý:")
        for name in sorted(flagged):
            print(f"  {name:32s} {' | '.join(sorted(flagged[name]))}")
    else:
        print("Tất cả sạch — không dataset nào có vấn đề.")

    # demo: chuẩn bị 1 dataset và in dtype/nhãn sau remap
    print("\n--- Demo prepare('wine', cv=0) ---")
    s, rep = load_prepared("wine", 0)
    print(f"kiểm tra thô: {rep.summary()}")
    print(f"sau prepare: X_train dtype={s.X_train.dtype}, "
          f"nhãn train unique={np.unique(s.y_train)}")
