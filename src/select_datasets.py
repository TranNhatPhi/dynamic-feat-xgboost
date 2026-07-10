"""
Chốt danh sách dataset cho thí nghiệm.

  - Loại `low-res-spect` (thiếu class trong train — phát hiện ở preprocessing).
  - DEV_SMALL : vài tập nhỏ để debug trên Mac (gần như miễn phí).
  - HARD_SET  : ~15 tập khó/nhiều chiều để làm nổi bật lợi thế tối ưu thời gian.

Đọc thông tin từ data/meta_datasets.csv và giao với các dataset thực có trong data/raw/.
"""

from __future__ import annotations

import csv
from pathlib import Path

from src.data_loader import DATA_ROOT, list_datasets

META_CSV = Path(__file__).resolve().parent.parent / "data" / "meta_datasets.csv"

# Loại vì lỗi split (xem docs/01)
EXCLUDE = {"low-res-spect"}

# Tập nhỏ để debug pipeline (nhanh, rẻ)
DEV_SMALL = ["wine", "iris", "seeds", "glass", "balloons"]

# Ứng viên "khó/nhiều chiều" do advisor gợi ý (sẽ giao với data thực có)
HARD_CANDIDATES = [
    "hill-valley", "statlog-image", "molec-biol-promoter", "monks-3",
    "vertebral-column-3clases", "arrhythmia", "libras", "plant-shape",
    "plant-margin", "plant-texture", "semeion", "musk-2",
    "pittsburg-bridges-MATERIAL", "pittsburg-bridges-TYPE", "conn-bench-sonar-mines-rocks",
]


def load_meta() -> dict[str, dict]:
    meta = {}
    with open(META_CSV, newline="") as f:
        for row in csv.DictReader(f):
            meta[row["dataset"]] = {
                "instances": int(row["instances"]),
                "classes": int(row["labels_classes"]),
                "features": int(row["features"]),
            }
    return meta


def available() -> set[str]:
    return set(list_datasets(DATA_ROOT)) - EXCLUDE


def resolve(names: list[str]) -> list[str]:
    """Chỉ giữ tên thực sự có trong data/raw (và không bị loại)."""
    have = available()
    return [n for n in names if n in have]


def rank_by_dimensionality(top: int = 15) -> list[str]:
    """Xếp hạng dataset theo số chiều (features) giảm dần — chọn top cái nhiều chiều nhất."""
    meta = load_meta()
    have = available()
    cand = [(n, meta[n]["features"]) for n in have if n in meta]
    cand.sort(key=lambda t: t[1], reverse=True)
    return [n for n, _ in cand[:top]]


def hard_set() -> list[str]:
    """HARD_SET = ứng viên advisor (có thực) + bù thêm bằng top-dimensionality cho đủ ~15."""
    picked = resolve(HARD_CANDIDATES)
    for n in rank_by_dimensionality(30):
        if len(picked) >= 15:
            break
        if n not in picked:
            picked.append(n)
    return picked[:15]


if __name__ == "__main__":
    meta = load_meta()
    have = available()
    print(f"Dataset khả dụng (đã trừ {EXCLUDE}): {len(have)}\n")

    def show(title, names):
        print(f"== {title} ({len(names)}) ==")
        for n in names:
            m = meta.get(n, {})
            print(f"  {n:32s} mẫu={m.get('instances','?'):>7} "
                  f"lớp={m.get('classes','?'):>3} feat={m.get('features','?'):>4}")
        print()

    show("DEV_SMALL (debug trên Mac)", resolve(DEV_SMALL))
    show("HARD_SET (thí nghiệm chính, nhiều chiều)", hard_set())

    # cảnh báo ứng viên advisor không tìm thấy
    missing = [n for n in HARD_CANDIDATES if n not in have]
    if missing:
        print(f"[chú ý] ứng viên không có trong data/raw: {missing}")
