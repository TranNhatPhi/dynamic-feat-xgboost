"""
Bộ nạp dữ liệu UCI cho dự án Dynamic Feat-XGBoost.

Sao chép ĐÚNG logic chia fold của tác giả gốc
(reference_code/featureEng/datasets/UCIdata.py) nhưng bỏ phụ thuộc `torch`,
chỉ dùng numpy — để kết quả so sánh với bài báo gốc là hợp lệ.

Mỗi dataset trong data/raw/<ten>/ gồm 4 file (delimiter = ','):
    <ten>_py.dat            ma trận đặc trưng X   (n_mau x n_features)
    labels_py.dat           nhãn y                (n_mau,)
    folds_py.dat            chỉ số fold           (n_mau x 4)  0=train-pool, 1=test
    validation_folds_py.dat chỉ số validation     (n_mau x 4)  1=validation

Với mỗi fold CV in {0,1,2,3}:
    test  = folds[:,CV] == 1
    val   = validation[:,CV] == 1
    train = (folds[:,CV] == 0) & (validation[:,CV] == 0)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Thư mục data mặc định: <repo>/data/raw
DATA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw"

N_FOLDS = 4  # bài gốc dùng 4-fold CV


@dataclass
class Split:
    """Một lần chia (train/val/test) cho một fold."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

    def shapes(self) -> str:
        return (
            f"train={self.X_train.shape} "
            f"val={self.X_val.shape} "
            f"test={self.X_test.shape}"
        )


class UCIDataset:
    """Đọc 4 file .dat của một dataset và tạo split theo fold."""

    def __init__(self, name: str, data_root: Path | str = DATA_ROOT):
        self.name = name
        self.root = Path(data_root) / name
        if not self.root.is_dir():
            raise FileNotFoundError(f"Không thấy thư mục dataset: {self.root}")

        data_file = sorted(self.root.glob(f"{name}*.dat"))[0]
        label_file = sorted(self.root.glob("label*.dat"))[0]
        val_file = sorted(self.root.glob("validation*.dat"))[0]
        fold_file = sorted(self.root.glob("folds*.dat"))[0]

        self.X = np.loadtxt(data_file, delimiter=",")
        self.y = np.loadtxt(label_file, delimiter=",").astype(np.int64)
        self.validation = np.loadtxt(val_file, delimiter=",")
        self.folds = np.loadtxt(fold_file, delimiter=",")

        if self.X.ndim == 1:  # dataset 1 đặc trưng -> ép về (n,1)
            self.X = self.X.reshape(-1, 1)
        self.n_folds = self.folds.shape[1]

    @property
    def n_classes(self) -> int:
        return int(np.unique(self.y).size)

    def get_split(self, cv: int) -> Split:
        if not 0 <= cv < self.n_folds:
            raise ValueError(f"cv phải trong [0,{self.n_folds}); nhận {cv}")

        folds_cv = self.folds[:, cv]
        val_cv = self.validation[:, cv]

        test_idx = np.where(folds_cv == 1)[0]
        val_idx = np.where(val_cv == 1)[0]
        train_idx = np.where((folds_cv == 0) & (val_cv == 0))[0]

        return Split(
            X_train=self.X[train_idx],
            y_train=self.y[train_idx],
            X_val=self.X[val_idx],
            y_val=self.y[val_idx],
            X_test=self.X[test_idx],
            y_test=self.y[test_idx],
        )


def list_datasets(data_root: Path | str = DATA_ROOT) -> list[str]:
    """Trả về tên tất cả dataset có trong data/raw/."""
    root = Path(data_root)
    return sorted(
        p.name
        for p in root.iterdir()
        if p.is_dir() and any(p.glob("folds*.dat"))
    )


def load(name: str, cv: int = 0, data_root: Path | str = DATA_ROOT) -> Split:
    """Tiện ích ngắn: load(name, cv) -> Split."""
    return UCIDataset(name, data_root).get_split(cv)


if __name__ == "__main__":
    names = list_datasets()
    print(f"Tìm thấy {len(names)} dataset trong {DATA_ROOT}")
    demo = "wine" if "wine" in names else names[0]
    ds = UCIDataset(demo)
    print(f"\nDataset '{demo}': {ds.X.shape[0]} mẫu, "
          f"{ds.X.shape[1]} đặc trưng, {ds.n_classes} lớp, {ds.n_folds} fold")
    for cv in range(ds.n_folds):
        s = ds.get_split(cv)
        print(f"  fold {cv}: {s.shapes()}")
