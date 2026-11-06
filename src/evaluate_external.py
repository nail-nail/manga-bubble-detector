"""Evaluate the detector on a hand-labelled external test set exported from Label Studio.

The labels have five classes (docs/LABELING_GUIDE.md) while the detector predicts one, so:

- The main score counts `speech` and `thought` boxes as bubbles.
- A prediction that matches a box of another class (narration, sfx, other) is neutral: it is neither
  a hit nor a false positive, because the training labels sometimes marked that kind of text too.
- Recall is also reported per class, to show which kinds of text the detector finds.

Usage:
    python -m src.evaluate_external --export data/external/onepiece/labelstudio/export.json
"""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

from src.dataset import read_gray
from src.evaluate import IOU_MATCH, assign_boxes, f1_from_counts
from src.inference import Predictor
from src.postprocess import Box

CLASSES = ("speech", "thought", "narration", "sfx", "other")
BUBBLE_CLASSES = ("speech", "thought")


def to_pixel_box(value: dict, width: int, height: int) -> list[float]:
    """Convert a Label Studio rectangle (percentages) to an [x, y, w, h] pixel box.

    The detector predicts upright boxes, so a rotated rectangle becomes the upright box around it.
    Label Studio rotates clockwise around the top-left corner.
    """
    x, y = value["x"] / 100 * width, value["y"] / 100 * height
    w, h = value["width"] / 100 * width, value["height"] / 100 * height
    angle = math.radians(value.get("rotation", 0))
    if not angle:
        return [x, y, w, h]
    cos, sin = math.cos(angle), math.sin(angle)
    xs = [x + cx * cos - cy * sin for cx, cy in ((0, 0), (w, 0), (w, h), (0, h))]
    ys = [y + cx * sin + cy * cos for cx, cy in ((0, 0), (w, 0), (w, h), (0, h))]
    return [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)]


def load_labelstudio(export_path: str | Path, images_dir: str | Path) -> list[dict]:
    """Read a Label Studio JSON export into pages with pixel boxes.

    Tasks with no submitted annotation are skipped, so a partly labelled project still works.
    """
    tasks = json.loads(Path(export_path).read_text(encoding="utf-8"))
    pages = []
    for task in tasks:
        annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not annotations:
            continue
        # Label Studio prefixes uploads with a random id: "81af392a-op_001.jpg".
        name = task.get("file_upload") or Path(task["data"]["image"]).name
        name = name.split("-", 1)[1] if "-" in name else name
        boxes, labels = [], []
        width = height = None
        for region in annotations[-1]["result"]:
            if region.get("type") != "rectanglelabels":
                continue
            width, height = region["original_width"], region["original_height"]
            value = region["value"]
            label = value["rectanglelabels"][0]
            if label not in CLASSES:
                raise ValueError(f"{name}: unknown class {label!r}")
            boxes.append(to_pixel_box(value, width, height))
            labels.append(label)
        pages.append({
            "id": Path(name).stem, "image": str(Path(images_dir) / name),
            "width": width, "height": height, "boxes": boxes, "labels": labels,
        })
    return sorted(pages, key=lambda page: page["id"])


def score_pages(pages: list[dict], predictions: dict[str, list[Box]], iou_threshold: float = IOU_MATCH) -> dict:
    """Score predictions against multi-class labels. See the module docstring for the rules."""
    tp = fp = neutral = 0
    found, total = Counter(), Counter()
    for page in pages:
        preds = predictions[page["id"]]
        pairs = assign_boxes(preds, page["boxes"], iou_threshold)
        total.update(page["labels"])
        found.update(page["labels"][target] for target in pairs.values())
        for pred_index in range(len(preds)):
            if pred_index not in pairs:
                fp += 1
            elif page["labels"][pairs[pred_index]] in BUBBLE_CLASSES:
                tp += 1
            else:
                neutral += 1
    fn = sum(total[c] for c in BUBBLE_CLASSES) - tp
    precision, recall, f1 = f1_from_counts(tp, fp, fn)
    return {
        "pages": len(pages),
        "bubbles": {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn},
        "neutral_matches": neutral,
        "recall_by_class": {
            c: {"found": found[c], "total": total[c], "recall": found[c] / total[c] if total[c] else None}
            for c in CLASSES
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--export", type=Path, required=True, help="Label Studio JSON export")
    parser.add_argument("--images", type=Path, default=Path("data/external/onepiece/images"))
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best.pt"))
    parser.add_argument("--name", default="onepiece", help="Used in the output file name")
    parser.add_argument("--out", type=Path, default=Path("reports"))
    args = parser.parse_args()

    pages = load_labelstudio(args.export, args.images)
    if not pages:
        parser.error("The export has no submitted annotations.")
    predictor = Predictor(args.checkpoint)
    predictions = {page["id"]: predictor.predict(read_gray(page["image"]))[0] for page in pages}

    summary = score_pages(pages, predictions)
    summary["checkpoint_version"] = predictor.version
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"metrics_unet_external_{args.name}.json"
    path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved to {path}")


if __name__ == "__main__":
    main()
