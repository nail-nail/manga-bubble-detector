"""Non-ML baseline: find bubbles with thresholding and contours only.

Idea: a speech bubble is a fairly large, compact white region that has dark text
inside it. This shows how much the U-Net actually adds over classic OpenCV.

Usage:
    python -m src.baseline --split val --tune
    python -m src.baseline --split test --final
"""

from __future__ import annotations

import argparse
import itertools
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from src.config import load_config
from src.dataset import letterbox, load_records, read_gray, scale_boxes
from src.evaluate import PageResult, score_page, summarize, summarize_by_source
from src.postprocess import Box


@dataclass(frozen=True)
class BaselineParams:
    white_threshold: int = 200  # Pixels brighter than this count as paper.
    min_area_frac: float = 0.001  # As a fraction of the canvas.
    max_area_frac: float = 0.15  # Larger white regions are background, not bubbles.
    min_solidity: float = 0.8  # Area over convex hull area. Bubbles are compact.
    min_ink_frac: float = 0.01  # Share of dark pixels inside. Bubbles contain text.


def detect(canvas: np.ndarray, params: BaselineParams) -> tuple[np.ndarray, list[Box]]:
    """Return a filled mask and boxes for a letterboxed grayscale page."""
    height, width = canvas.shape
    white = (canvas > params.white_threshold).astype(np.uint8)
    # RETR_EXTERNAL plus FILLED ignores the holes that text punches in a bubble.
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    mask = np.zeros_like(canvas, dtype=np.uint8)
    boxes: list[Box] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not params.min_area_frac * height * width <= area <= params.max_area_frac * height * width:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        # White regions touching the edge are page margin or letterbox padding.
        if x == 0 or y == 0 or x + w >= width or y + h >= height:
            continue
        hull_area = cv2.contourArea(cv2.convexHull(contour))
        if hull_area == 0 or area / hull_area < params.min_solidity:
            continue

        region = np.zeros_like(canvas, dtype=np.uint8)
        cv2.drawContours(region, [contour], -1, 1, cv2.FILLED)
        ink = float((canvas[region == 1] <= params.white_threshold).mean())
        if ink < params.min_ink_frac:
            continue

        mask[region == 1] = 1
        boxes.append(Box(x, y, w, h, 1.0))
    return mask.astype(bool), boxes


def load_pages(records: list[dict], image_size: int) -> list[dict]:
    pages = []
    for record in tqdm(records, desc="Loading", leave=False):
        canvas, info = letterbox(read_gray(record["image"]), image_size)
        gt_mask, _ = letterbox(read_gray(record["mask"]), image_size, cv2.INTER_NEAREST, pad_value=0)
        pages.append(
            {
                "id": record["id"],
                "canvas": canvas,
                "mask": gt_mask > 127,
                "boxes": scale_boxes(np.array(record["boxes"], dtype=np.float32), info),
            }
        )
    return pages


def score_baseline(pages: list[dict], params: BaselineParams) -> list[PageResult]:
    results = []
    for page in pages:
        mask, boxes = detect(page["canvas"], params)
        results.append(score_page(page["id"], mask, boxes, page["mask"], page["boxes"]))
    return results


def tune(pages: list[dict]) -> BaselineParams:
    grid = itertools.product(
        (180, 200, 225), (0.0005, 0.001, 0.002), (0.1, 0.2), (0.7, 0.85), (0.0, 0.01, 0.03)
    )
    best, best_f1 = BaselineParams(), -1.0
    for values in tqdm(list(grid), desc="Tuning", leave=False):
        params = BaselineParams(*values)
        f1 = summarize(score_baseline(pages, params))["f1"]
        if f1 > best_f1:
            best, best_f1 = params, f1
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--final", action="store_true")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()
    if args.split == "test" and not args.final:
        parser.error("The test set is for the final evaluation only. Pass --final if this is it.")
    if args.split == "test" and args.tune:
        parser.error("Tuning on the test set would leak it. Use --split val.")

    config = load_config(args.config)
    records = load_records(config["data"]["splits_dir"], args.split)
    pages = load_pages(records, config["data"]["image_size"])

    args.out.mkdir(parents=True, exist_ok=True)
    params_path = args.out / "baseline_params.json"
    if args.tune:
        params = tune(pages)
        params_path.write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")
    elif params_path.exists():
        params = BaselineParams(**json.loads(params_path.read_text(encoding="utf-8")))
    else:
        params = BaselineParams()
        print("Warning: no tuned baseline parameters found. Using defaults.")

    results = score_baseline(pages, params)
    summary = summarize(results)
    summary["by_source"] = summarize_by_source(results, records)

    timings = []
    for page in pages[:30]:
        started = time.perf_counter()
        detect(page["canvas"], params)
        timings.append((time.perf_counter() - started) * 1000)
    summary["cpu_ms_per_page"] = round(statistics.median(timings), 1)
    summary["parameters"] = 0
    summary["baseline_params"] = asdict(params)

    (args.out / f"metrics_baseline_{args.split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
