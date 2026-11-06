"""Convert raw labels to image + mask pairs and write leak-free train/val/test splits.

Why this doesn't use the dataset's own splits (see docs/DATA_NOTES.md for evidence):

* Its splits scatter every scan source across train, val and test, so test pages
  share a series and art style with training pages.
* About half of its pages are flipped or rotated copies of other pages, and some
  of those copies cross the split boundary.

There is no title metadata, so a page's image resolution stands in for its source.
Whole sources are held out, and near-duplicates are never allowed to cross a split.

Usage:
    python -m src.prepare_data
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imagehash
import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from src.config import load_config

RAW_SPLITS = ("train", "val", "test")

# Which whole sources are held out. Everything else trains.
HOLDOUT = {
    "screens_3840": "test",
    "portrait_764": "test",
    "scan_1044": "test",
    "portrait_2150": "val",
}
# Labels here are unreliable: in a sample of 30 "empty" pages, about a third had visible
# unlabelled bubbles. In other sources, empty pages checked by eye really had none, so
# those are kept.
DROP_EMPTY_SOURCES = {"square_640"}

# Hamming distance on a 64-bit pHash. At 14, unrelated pages start chaining into one
# giant group (1,299 pages), so 10 is about as loose as this check can safely go.
DUPLICATE_MAX_DISTANCE = 10

SPLIT_PRIORITY = {"test": 0, "val": 1, "train": 2}


def source_of(width: int, height: int) -> str:
    """Infer a page's scan source from its resolution."""
    if (width, height) == (640, 640):
        return "square_640"
    if (width, height) == (3840, 2160):
        return "screens_3840"
    if height == 1200 and 750 <= width <= 780:
        return "portrait_764"
    if height == 1200 and 830 <= width <= 860:
        return "portrait_844"
    if 2130 <= height <= 2170 and width < 2000:
        return "portrait_2150"
    if 1020 <= height <= 1070 and 1350 <= width <= 1500:
        return "scan_1044"
    return "other"


def parse_label_line(line: str, width: int, height: int) -> tuple[str, np.ndarray] | None:
    """Parse a YOLO line into ("box" or "polygon", N x 2 pixel points).

    Five values means a detection box: class cx cy w h. More than that means a
    segmentation polygon: class x1 y1 x2 y2 ... Handling both lets a mask dataset
    replace this one without changes anywhere else in the pipeline.
    """
    parts = line.split()
    if not parts:
        return None
    values = [float(v) for v in parts[1:]]
    if len(values) == 4:
        cx, cy, w, h = values
        points = np.array(
            [
                [(cx - w / 2) * width, (cy - h / 2) * height],
                [(cx + w / 2) * width, (cy - h / 2) * height],
                [(cx + w / 2) * width, (cy + h / 2) * height],
                [(cx - w / 2) * width, (cy + h / 2) * height],
            ]
        )
        return "box", points
    if len(values) >= 6 and len(values) % 2 == 0:
        points = np.array(values).reshape(-1, 2) * [width, height]
        return "polygon", points
    raise ValueError(f"Unrecognized label line: {line!r}")


def rasterize(lines: list[str], width: int, height: int) -> tuple[np.ndarray, list[list[int]], str | None]:
    """Build a binary mask and one [x, y, w, h] box per instance."""
    mask = np.zeros((height, width), dtype=np.uint8)
    boxes: list[list[int]] = []
    kind = None
    for line in lines:
        parsed = parse_label_line(line, width, height)
        if parsed is None:
            continue
        kind, points = parsed
        points = np.round(points).astype(np.int32)
        points[:, 0] = np.clip(points[:, 0], 0, width)
        points[:, 1] = np.clip(points[:, 1], 0, height)
        x0, y0 = points.min(axis=0)
        x1, y1 = points.max(axis=0)
        if x1 <= x0 or y1 <= y0:
            continue
        cv2.fillPoly(mask, [points], 255)
        boxes.append([int(x0), int(y0), int(x1 - x0), int(y1 - y0)])
    return mask, boxes, kind


def page_hashes(path: Path) -> np.ndarray:
    """pHashes of a page in every orientation. pHash shrinks to a square, so it also survives stretching."""
    with Image.open(path) as image:
        gray = image.convert("L")
        variants = (
            gray,
            ImageOps.mirror(gray),
            ImageOps.flip(gray),
            gray.rotate(180),
            gray.rotate(90, expand=True),
            gray.rotate(270, expand=True),
        )
        return np.array([np.array(imagehash.phash(v).hash, dtype=bool).ravel() for v in variants])


def duplicate_groups(hashes: np.ndarray, max_distance: int = DUPLICATE_MAX_DISTANCE) -> list[int]:
    """Union-find over near-duplicates. hashes is N x orientations x 64."""
    count = len(hashes)
    parent = list(range(count))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    upright = hashes[:, 0]
    for orientation in range(hashes.shape[1]):
        distances = (upright[:, None, :] != hashes[:, orientation][None, :, :]).sum(axis=-1)
        for i, j in np.argwhere(distances <= max_distance):
            if i != j:
                parent[find(int(i))] = find(int(j))
    return [find(i) for i in range(count)]


def resolve_duplicate_conflicts(splits: list[str], groups: list[int]) -> list[bool]:
    """Return a keep-flag per page, so no duplicate group spans two splits.

    The group stays in its most protected split (test, then val, then train), and
    its copies elsewhere are dropped. The test set is never the side that gets cut.
    """
    best: dict[int, str] = {}
    for split, group in zip(splits, groups):
        if group not in best or SPLIT_PRIORITY[split] < SPLIT_PRIORITY[best[group]]:
            best[group] = split
    return [split == best[group] for split, group in zip(splits, groups)]


@dataclass
class Page:
    id: str
    image: Path
    width: int
    height: int
    lines: list[str]
    source: str
    split: str = field(default="train")


def load_raw_pages(raw_dir: Path) -> list[Page]:
    pages = []
    for raw_split in RAW_SPLITS:
        image_paths = sorted((raw_dir / "images" / raw_split).glob("*.jpg"))
        if not image_paths:
            raise FileNotFoundError(
                f"No images in {raw_dir / 'images' / raw_split}. Run `python -m src.download_data` first."
            )
        for path in image_paths:
            label_path = raw_dir / "labels" / raw_split / f"{path.stem}.txt"
            text = label_path.read_text(encoding="utf-8") if label_path.exists() else ""
            with Image.open(path) as image:
                width, height = image.size
            source = source_of(width, height)
            pages.append(
                Page(
                    id=f"{raw_split}_{path.stem}",
                    image=path,
                    width=width,
                    height=height,
                    lines=[line for line in text.splitlines() if line.strip()],
                    source=source,
                    split=HOLDOUT.get(source, "train"),
                )
            )
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/manga_bubbles_detect"))
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    config = load_config(args.config)
    masks_dir = Path(config["data"]["processed_dir"]) / "masks"
    splits_dir = Path(config["data"]["splits_dir"])
    masks_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    pages = load_raw_pages(args.raw)
    report: dict = {"raw_pages": len(pages), "raw_by_source": dict(Counter(p.source for p in pages))}

    kept = [p for p in pages if p.lines or p.source not in DROP_EMPTY_SOURCES]
    report["dropped_unreliable_empty"] = len(pages) - len(kept)

    hashes = np.array([page_hashes(p.image) for p in tqdm(kept, desc="Hashing")])
    groups = duplicate_groups(hashes)
    keep_flags = resolve_duplicate_conflicts([p.split for p in kept], groups)
    report["dropped_cross_split_duplicates"] = len(kept) - sum(keep_flags)
    report["pages_with_a_duplicate"] = sum(c for c in Counter(groups).values() if c > 1)

    records: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for page, group, keep in tqdm(list(zip(kept, groups, keep_flags)), desc="Writing masks"):
        if not keep:
            continue
        mask, boxes, kind = rasterize(page.lines, page.width, page.height)
        mask_path = masks_dir / f"{page.id}.png"
        cv2.imwrite(str(mask_path), mask)
        records[page.split].append(
            {
                "id": page.id,
                "image": page.image.as_posix(),
                "mask": mask_path.as_posix(),
                "width": page.width,
                "height": page.height,
                "boxes": boxes,
                "source": page.source,
                "annotation": kind,
                "duplicate_group": group,
            }
        )

    for split, items in records.items():
        (splits_dir / f"{split}.json").write_text(json.dumps(items), encoding="utf-8")

    total = sum(len(items) for items in records.values())
    report["splits"] = {
        split: {
            "pages": len(items),
            "share": round(len(items) / total, 3),
            "bubbles": sum(len(r["boxes"]) for r in items),
            "empty_pages": sum(not r["boxes"] for r in items),
            "by_source": dict(Counter(r["source"] for r in items)),
        }
        for split, items in records.items()
    }
    (splits_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
