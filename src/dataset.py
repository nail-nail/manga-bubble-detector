"""Page preprocessing, augmentation and the PyTorch Dataset.

Records come from data/splits/<split>.json, written by src/prepare_data.py. Each
record points to a grayscale-readable page and a binary mask, plus one box per
bubble, so nothing here depends on how the raw labels were formatted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# Paper is white, so padding with white looks like more page rather than a hard edge.
PAD_VALUE = 255
NORM_MEAN = 0.5
NORM_STD = 0.5


@dataclass(frozen=True)
class LetterboxInfo:
    """What is needed to map network coordinates back to the original page."""

    scale: float
    pad_x: int
    pad_y: int
    orig_w: int
    orig_h: int


def letterbox(
    image: np.ndarray, size: int, interpolation: int | None = None, pad_value: int = PAD_VALUE
) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize to fit inside size x size while keeping aspect ratio, then pad.

    Stretching a tall page into a square would change the shape of the bubbles
    from what the model sees on other pages, so the page is padded instead.
    """
    orig_h, orig_w = image.shape[:2]
    scale = size / max(orig_w, orig_h)
    new_w = max(1, round(orig_w * scale))
    new_h = max(1, round(orig_h * scale))
    if interpolation is None:
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)

    pad_x = (size - new_w) // 2
    pad_y = (size - new_h) // 2
    canvas = np.full((size, size), pad_value, dtype=image.dtype)
    canvas[pad_y : pad_y + new_h, pad_x : pad_x + new_w] = resized
    return canvas, LetterboxInfo(scale, pad_x, pad_y, orig_w, orig_h)


def to_tensor(gray: np.ndarray) -> torch.Tensor:
    """H x W uint8 to a 1 x H x W float tensor normalized to [-1, 1]."""
    tensor = torch.from_numpy(gray.astype(np.float32) / 255.0)
    return ((tensor - NORM_MEAN) / NORM_STD).unsqueeze(0)


def preprocess_page(gray: np.ndarray, size: int) -> tuple[torch.Tensor, LetterboxInfo]:
    """The exact preprocessing used by validation, evaluation and the API."""
    canvas, info = letterbox(gray, size)
    return to_tensor(canvas), info


def scale_boxes(boxes: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Map N x 4 [x, y, w, h] boxes from the page to network coordinates."""
    scaled = boxes.astype(np.float32).reshape(-1, 4) * info.scale
    scaled[:, 0] += info.pad_x
    scaled[:, 1] += info.pad_y
    return scaled


def augment(
    image: np.ndarray, mask: np.ndarray, cfg: dict, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Apply the same geometric change to both arrays, and photometric changes to the image."""
    if rng.random() < cfg["hflip_prob"]:
        # Safe here because the model finds bubble shapes and never reads the text.
        image, mask = image[:, ::-1].copy(), mask[:, ::-1].copy()

    degrees = cfg["rotate_degrees"]
    if degrees > 0:
        height, width = image.shape
        matrix = cv2.getRotationMatrix2D(
            (width / 2, height / 2), float(rng.uniform(-degrees, degrees)), 1.0
        )
        image = cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=PAD_VALUE
        )
        # Nearest neighbour keeps the mask strictly binary.
        mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST, borderValue=0)

    if rng.random() < cfg["crop_prob"]:
        height, width = image.shape
        side = float(rng.uniform(*cfg["crop_scale"]))
        crop_w, crop_h = max(1, int(width * side)), max(1, int(height * side))
        x0 = int(rng.integers(0, width - crop_w + 1))
        y0 = int(rng.integers(0, height - crop_h + 1))
        image = image[y0 : y0 + crop_h, x0 : x0 + crop_w]
        mask = mask[y0 : y0 + crop_h, x0 : x0 + crop_w]

    contrast = 1.0 + float(rng.uniform(-cfg["contrast"], cfg["contrast"]))
    brightness = 255.0 * float(rng.uniform(-cfg["brightness"], cfg["brightness"]))
    adjusted = (image.astype(np.float32) - 127.5) * contrast + 127.5 + brightness
    image = np.clip(adjusted, 0, 255).astype(np.uint8)
    return image, mask


def read_gray(path: str | Path) -> np.ndarray:
    # np.fromfile + imdecode also copes with non-ASCII Windows paths; cv2.imread does not.
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def load_records(splits_dir: str | Path, split: str) -> list[dict]:
    path = Path(splits_dir) / f"{split}.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.prepare_data` first.")
    return json.loads(path.read_text(encoding="utf-8"))


class BubbleDataset(Dataset):
    """Yields image, mask, id and, when not augmenting, ground-truth boxes."""

    def __init__(
        self,
        records: list[dict],
        image_size: int,
        augment_cfg: dict | None = None,
        root: str | Path = ".",
    ) -> None:
        self.records = records
        self.image_size = image_size
        self.augment_cfg = augment_cfg
        self.root = Path(root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        image = read_gray(self.root / record["image"])
        mask = read_gray(self.root / record["mask"])

        if self.augment_cfg is not None:
            # Drawn from torch's RNG, which PyTorch seeds separately in each worker.
            # So every page gets a new augmentation each epoch, workers never repeat
            # each other, and a fixed global seed still reproduces the whole run.
            rng = np.random.default_rng(int(torch.randint(0, 2**31 - 1, (1,)).item()))
            image, mask = augment(image, mask, self.augment_cfg, rng)

        canvas, info = letterbox(image, self.image_size)
        mask_canvas, _ = letterbox(mask, self.image_size, cv2.INTER_NEAREST, pad_value=0)

        # Boxes stop matching once a page is augmented. They are only used in evaluation.
        boxes = np.zeros((0, 4), dtype=np.float32)
        if self.augment_cfg is None:
            boxes = scale_boxes(np.array(record["boxes"], dtype=np.float32), info)

        return {
            "image": to_tensor(canvas),
            "mask": torch.from_numpy((mask_canvas > 127).astype(np.float32)).unsqueeze(0),
            "boxes": torch.from_numpy(boxes),
            "id": record["id"],
        }


def collate(batch: list[dict]) -> dict:
    """Stack the tensors. Boxes stay as a list because each page has a different count."""
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "mask": torch.stack([item["mask"] for item in batch]),
        "boxes": [item["boxes"] for item in batch],
        "id": [item["id"] for item in batch],
    }
