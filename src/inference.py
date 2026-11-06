"""Load a trained checkpoint and run the full page -> boxes pipeline."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from src.dataset import preprocess_page
from src.model import UNet
from src.postprocess import Box, PostprocessParams, boxes_to_original, mask_to_boxes


def tuned_params_path(checkpoint: str | Path) -> Path:
    """Where `evaluate --tune` stores the postprocessing values for a checkpoint."""
    checkpoint = Path(checkpoint)
    return checkpoint.with_name(f"{checkpoint.stem}.postprocess.json")


def load_postprocess_params(checkpoint: str | Path, config: dict) -> PostprocessParams:
    path = tuned_params_path(checkpoint)
    if path.exists():
        return PostprocessParams(**json.loads(path.read_text(encoding="utf-8")))
    return PostprocessParams(**config["postprocess"])


class Predictor:
    def __init__(self, checkpoint: str | Path, device: str = "cpu") -> None:
        checkpoint = Path(checkpoint)
        self.device = torch.device(device)
        # weights_only=True refuses to unpickle arbitrary objects, so loading a
        # checkpoint someone else published can't execute code.
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        config = state["config"]
        self.image_size = config["data"]["image_size"]
        self.model = UNet(
            in_channels=config["model"]["in_channels"],
            base_channels=config["model"]["base_channels"],
        ).to(self.device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.params = load_postprocess_params(checkpoint, config)
        self.version = hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:12]

    @torch.no_grad()
    def predict(self, gray: np.ndarray) -> tuple[list[Box], float]:
        """Return boxes in original page coordinates, and the time taken in ms."""
        started = time.perf_counter()
        tensor, info = preprocess_page(gray, self.image_size)
        logits = self.model(tensor.unsqueeze(0).to(self.device))
        prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
        boxes = boxes_to_original(
            mask_to_boxes(prob, self.params),
            info.scale, info.pad_x, info.pad_y, info.orig_w, info.orig_h,
        )
        return boxes, (time.perf_counter() - started) * 1000.0
