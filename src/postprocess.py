"""Turn a probability mask into one bounding box per bubble."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Box:
    x: int
    y: int
    w: int
    h: int
    score: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PostprocessParams:
    threshold: float = 0.5
    open_kernel: int = 3  # Morphological opening size in px. 0 or 1 disables it.
    min_area: int = 100  # Components smaller than this (px) are treated as noise.
    erode_px: int = 0  # Shrink before labelling to split touching bubbles.


def _ellipse(size: int) -> np.ndarray:
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))


def mask_to_boxes(prob: np.ndarray, params: PostprocessParams = PostprocessParams()) -> list[Box]:
    """Convert an H x W probability map in [0, 1] to boxes in the same pixel grid.

    Steps: threshold, opening, optional erosion, connected components, drop small
    components, then take each component's bounding box. A box's score is the
    mean probability over the pixels of its component.
    """
    if prob.ndim != 2:
        raise ValueError(f"Expected a 2D probability map, got shape {prob.shape}")

    binary = (prob > params.threshold).astype(np.uint8)
    if params.open_kernel > 1:
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, _ellipse(params.open_kernel))
    if params.erode_px > 0:
        binary = cv2.erode(binary, _ellipse(2 * params.erode_px + 1))

    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=4)

    height, width = prob.shape
    boxes: list[Box] = []
    for label in range(1, count):  # Label 0 is the background.
        if stats[label, cv2.CC_STAT_AREA] < params.min_area:
            continue
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])

        # Erosion shrank the component on every side, so grow the box back.
        x0 = max(x - params.erode_px, 0)
        y0 = max(y - params.erode_px, 0)
        x1 = min(x + w + params.erode_px, width)
        y1 = min(y + h + params.erode_px, height)

        score = float(prob[labels == label].mean())
        boxes.append(Box(x0, y0, x1 - x0, y1 - y0, round(score, 4)))

    boxes.sort(key=lambda box: box.score, reverse=True)
    return boxes


def boxes_to_original(
    boxes: list[Box], scale: float, pad_x: int, pad_y: int, orig_w: int, orig_h: int
) -> list[Box]:
    """Undo the letterbox so box coordinates refer to the uploaded page."""
    result: list[Box] = []
    for box in boxes:
        x0 = int(round((box.x - pad_x) / scale))
        y0 = int(round((box.y - pad_y) / scale))
        x1 = int(round((box.x + box.w - pad_x) / scale))
        y1 = int(round((box.y + box.h - pad_y) / scale))
        x0, x1 = max(x0, 0), min(x1, orig_w)
        y0, y1 = max(y0, 0), min(y1, orig_h)
        if x1 > x0 and y1 > y0:  # A box lying fully inside the padding vanishes.
            result.append(Box(x0, y0, x1 - x0, y1 - y0, box.score))
    return result


def draw_boxes(image_bgr: np.ndarray, boxes: list[Box]) -> np.ndarray:
    """Return a copy of the image with boxes and scores drawn on it."""
    canvas = image_bgr.copy()
    thickness = max(2, round(max(canvas.shape[:2]) / 500))
    font_scale = max(0.5, max(canvas.shape[:2]) / 1600)
    for box in boxes:
        cv2.rectangle(
            canvas, (box.x, box.y), (box.x + box.w, box.y + box.h), (0, 0, 255), thickness
        )
        cv2.putText(
            canvas,
            f"{box.score:.2f}",
            (box.x, max(box.y - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 255),
            max(1, thickness // 2),
            cv2.LINE_AA,
        )
    return canvas
