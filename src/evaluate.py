"""Evaluate a checkpoint: Dice, IoU, and per-bubble precision/recall at IoU 0.5.

Usage:
    python -m src.evaluate --checkpoint checkpoints/best.pt --split val --tune
    python -m src.evaluate --checkpoint checkpoints/best.pt --split test --final

`--tune` chooses postprocessing values on the validation set. The test set needs
an explicit `--final`, because each look at it makes its score less trustworthy.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from src.config import load_config
from src.dataset import BubbleDataset, load_records, read_gray
from src.inference import Predictor, tuned_params_path
from src.model import count_parameters
from src.postprocess import Box, PostprocessParams, mask_to_boxes
from src.train import build_loader, build_model, pick_device

IOU_MATCH = 0.5


def pixel_scores(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Dice and IoU of two boolean masks. Two empty masks agree perfectly."""
    intersection = float(np.logical_and(pred, target).sum())
    pred_sum, target_sum = float(pred.sum()), float(target.sum())
    union = pred_sum + target_sum - intersection
    if union == 0:
        return 1.0, 1.0
    return 2 * intersection / (pred_sum + target_sum), intersection / union


def box_iou(a, b) -> float:
    """IoU of two [x, y, w, h] boxes."""
    overlap_w = min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0])
    overlap_h = min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1])
    if overlap_w <= 0 or overlap_h <= 0:
        return 0.0
    overlap = overlap_w * overlap_h
    return float(overlap / (a[2] * a[3] + b[2] * b[3] - overlap))


def assign_boxes(preds: list[Box], targets, iou_threshold: float = IOU_MATCH) -> dict[int, int]:
    """Greedy one-to-one matching, highest score first. Returns {pred index: target index}.

    Each ground-truth box can be claimed once, so two predictions on one bubble
    give one match and one unmatched prediction.
    """
    unmatched = list(range(len(targets)))
    pairs = {}
    order = sorted(range(len(preds)), key=lambda i: preds[i].score, reverse=True)
    for pred_index in order:
        pred = preds[pred_index]
        coords = (pred.x, pred.y, pred.w, pred.h)
        best_iou, best_index = 0.0, None
        for index in unmatched:
            iou = box_iou(coords, targets[index])
            if iou > best_iou:
                best_iou, best_index = iou, index
        if best_index is not None and best_iou >= iou_threshold:
            pairs[pred_index] = best_index
            unmatched.remove(best_index)
    return pairs


def match_boxes(preds: list[Box], targets, iou_threshold: float = IOU_MATCH) -> tuple[int, int, int]:
    """Greedy one-to-one matching, highest score first. Returns TP, FP, FN."""
    true_positives = len(assign_boxes(preds, targets, iou_threshold))
    return true_positives, len(preds) - true_positives, len(targets) - true_positives


def f1_from_counts(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


@dataclass
class PageResult:
    id: str
    dice: float
    iou: float
    tp: int
    fp: int
    fn: int

    @property
    def f1(self) -> float:
        if self.tp + self.fp + self.fn == 0:
            return 1.0  # No bubbles on the page, and none predicted.
        return f1_from_counts(self.tp, self.fp, self.fn)[2]


def score_page(page_id: str, pred_mask, pred_boxes, gt_mask, gt_boxes) -> PageResult:
    dice, iou = pixel_scores(pred_mask, gt_mask)
    tp, fp, fn = match_boxes(pred_boxes, gt_boxes)
    return PageResult(page_id, dice, iou, tp, fp, fn)


def summarize(results: list[PageResult]) -> dict:
    """Dice and IoU are averaged per page. P/R/F1 pool all bubbles across pages."""
    tp = sum(r.tp for r in results)
    fp = sum(r.fp for r in results)
    fn = sum(r.fn for r in results)
    precision, recall, f1 = f1_from_counts(tp, fp, fn)
    return {
        "pages": len(results),
        "dice": float(np.mean([r.dice for r in results])) if results else 0.0,
        "iou": float(np.mean([r.iou for r in results])) if results else 0.0,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def summarize_by_source(results: list[PageResult], records: list[dict]) -> dict:
    """The same metrics per scan source, so one easy or hard source can't hide in the average."""
    source_of_page = {record["id"]: record["source"] for record in records}
    grouped: dict[str, list[PageResult]] = {}
    for result in results:
        grouped.setdefault(source_of_page[result.id], []).append(result)
    return {source: summarize(items) for source, items in sorted(grouped.items())}


@torch.no_grad()
def predict_split(model, loader, device) -> list[dict]:
    model.eval()
    samples = []
    for batch in tqdm(loader, desc="Predicting", leave=False):
        probs = torch.sigmoid(model(batch["image"].to(device)))[:, 0].cpu().numpy()
        masks = batch["mask"][:, 0].numpy().astype(bool)
        for i, page_id in enumerate(batch["id"]):
            samples.append(
                {"id": page_id, "prob": probs[i], "mask": masks[i], "boxes": batch["boxes"][i].numpy()}
            )
    return samples


def score_samples(samples: list[dict], params: PostprocessParams) -> list[PageResult]:
    return [
        score_page(
            s["id"], s["prob"] > params.threshold, mask_to_boxes(s["prob"], params), s["mask"], s["boxes"]
        )
        for s in samples
    ]


def tune_postprocess(samples: list[dict], grid: dict) -> tuple[PostprocessParams, dict]:
    """Pick the values with the best instance F1. Call this on validation data only."""
    best_params, best_summary = None, None
    combos = list(itertools.product(*(grid[k] for k in ("threshold", "open_kernel", "min_area", "erode_px"))))
    for threshold, open_kernel, min_area, erode_px in tqdm(combos, desc="Tuning", leave=False):
        params = PostprocessParams(threshold, open_kernel, min_area, erode_px)
        summary = summarize(score_samples(samples, params))
        if best_summary is None or summary["f1"] > best_summary["f1"]:
            best_params, best_summary = params, summary
    return best_params, best_summary


def measure_cpu_latency(checkpoint: Path, records: list[dict], pages: int = 30) -> float:
    """Median ms per page for the full pipeline on CPU, the setting the API runs in."""
    predictor = Predictor(checkpoint, device="cpu")
    timings = []
    for record in records[:pages]:
        _, elapsed = predictor.predict(read_gray(record["image"]))
        timings.append(elapsed)
    # The first run includes one-off allocations, so it isn't representative.
    return statistics.median(timings[1:] if len(timings) > 1 else timings)


def save_worst_pages(
    results: list[PageResult], samples: list[dict], params: PostprocessParams, out_dir: Path, count: int = 10
) -> list[PageResult]:
    """Write overlays of the worst pages for error analysis: green = truth, red = prediction."""
    out_dir.mkdir(parents=True, exist_ok=True)
    by_id = {s["id"]: s for s in samples}
    worst = sorted(results, key=lambda r: (r.f1, r.dice))[:count]
    with open(out_dir / "worst.csv", "w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["rank", "id", "f1", "dice", "tp", "fp", "fn", "category"])
        for rank, result in enumerate(worst, start=1):
            writer.writerow(
                [rank, result.id, f"{result.f1:.3f}", f"{result.dice:.3f}",
                 result.tp, result.fp, result.fn, ""]
            )
            sample = by_id[result.id]
            image = (np.clip(sample["image"] * 0.5 + 0.5, 0, 1) * 255).astype(np.uint8)
            canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            for x, y, w, h in sample["boxes"]:
                cv2.rectangle(canvas, (int(x), int(y)), (int(x + w), int(y + h)), (0, 170, 0), 2)
            for box in mask_to_boxes(sample["prob"], params):
                cv2.rectangle(canvas, (box.x, box.y), (box.x + box.w, box.y + box.h), (0, 0, 255), 2)
            cv2.imwrite(str(out_dir / f"{rank:02d}_{result.id}.png"), canvas)
    return worst


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--tune", action="store_true", help="Tune postprocessing (val only)")
    parser.add_argument("--final", action="store_true", help="Required to evaluate the test split")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()

    if args.split == "test" and not args.final:
        parser.error("The test set is for the final evaluation only. Pass --final if this is it.")
    if args.split == "test" and args.tune:
        parser.error("Tuning on the test set would leak it into the model selection. Use --split val.")

    device = pick_device(args.device)
    state = torch.load(args.checkpoint, map_location=device, weights_only=True)
    # Use the checkpoint's own config so preprocessing always matches training.
    config = state.get("config") or load_config(args.config)
    model = build_model(config).to(device)
    model.load_state_dict(state["model"])

    records = load_records(config["data"]["splits_dir"], args.split)
    dataset = BubbleDataset(records, config["data"]["image_size"])
    loader = build_loader(dataset, config["train"]["batch_size"], config["data"]["num_workers"], train=False)
    samples = predict_split(model, loader, device)

    if args.tune:
        # The grid is a postprocessing choice, not part of training, so read it from the
        # current config file. That lets a finished checkpoint be retuned on a wider grid.
        params, _ = tune_postprocess(samples, load_config(args.config)["tune_grid"])
        path = tuned_params_path(args.checkpoint)
        path.write_text(json.dumps(asdict(params), indent=2), encoding="utf-8")
        print(f"Tuned on val: {params}\nSaved to {path}")
    elif tuned_params_path(args.checkpoint).exists():
        params = PostprocessParams(**json.loads(tuned_params_path(args.checkpoint).read_text(encoding="utf-8")))
    else:
        params = PostprocessParams(**config["postprocess"])
        print("Warning: no tuned postprocessing found. Using config defaults.")

    results = score_samples(samples, params)
    summary = summarize(results)
    summary["by_source"] = summarize_by_source(results, records)
    summary["parameters"] = count_parameters(model)
    summary["cpu_ms_per_page"] = round(measure_cpu_latency(args.checkpoint, records), 1)
    summary["postprocess"] = asdict(params)

    # Overlays need the input image, which predict_split leaves out to save memory.
    worst_ids = {r.id for r in sorted(results, key=lambda r: (r.f1, r.dice))[:10]}
    for index, record in enumerate(records):
        if record["id"] in worst_ids:
            sample = next(s for s in samples if s["id"] == record["id"])
            sample["image"] = dataset[index]["image"][0].numpy()
    save_worst_pages(results, samples, params, args.out / f"worst_{args.split}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / f"metrics_unet_{args.split}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
