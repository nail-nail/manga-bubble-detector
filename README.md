# Manga Speech-Bubble Detector

Finds speech bubbles on a manga page and returns one bounding box per bubble. It uses a U-Net written
from scratch in PyTorch, with a hand-written training loop and evaluation, served through FastAPI and
Docker.

This is the first stage of a planned open-source manga translator: **detect** → OCR → translate →
typeset.

<!-- TODO: add assets/demo.png. Use a page you have the rights to share, not a dataset page. -->

## How it works

```
manga page (PNG/JPG)
   │
   ▼
preprocess ── grayscale, letterbox to 512×512, normalize
   │
   ▼
U-Net ── probability mask, 1×512×512
   │
   ▼
postprocess ── threshold → opening → connected components
   │           → remove small components → one box per bubble
   ▼
output ── JSON [{x, y, w, h, score}] or an annotated PNG
```

**Why segmentation rather than direct detection?** Bubbles aren't rectangular (ovals, jagged edges,
tails), and a U-Net is small enough to write from scratch and explain line by line. Boxes still come
out of postprocessing. With this dataset's box labels, the first of those benefits isn't realized
yet: see [Limitations](#limitations).

**Why BCE + Dice?** Bubbles cover only about 15% of each page, so a pixel-averaged loss can look good
while finding nothing. Dice scores overlap with the foreground regardless of how much background there
is, and BCE provides stable gradients.

## Results

> **Not yet run.** The table is filled in after GPU training and the final test evaluation.
> Run `python -m src.evaluate --checkpoint checkpoints/best.pt --split test --final`.

| Model | Dice | IoU | Precision @0.5 | Recall @0.5 | F1 | Params | CPU ms/page |
|---|---|---|---|---|---|---|---|
| Baseline (OpenCV, no ML) | | | | | | 0 | |
| U-Net (from scratch) | | | | | | | |

Test set: 444 pages and 4,361 bubbles from sources the model never trained on. Precision, recall and
F1 pool all bubbles, with a match requiring box IoU ≥ 0.5.

For reference, the tuned OpenCV baseline scores F1 0.43 on **validation** (precision 0.79, recall 0.30,
9 ms/page). It is precise but misses most bubbles.

## Data

Uses [`PSImera/manga_bubbles_detect`](https://huggingface.co/datasets/PSImera/manga_bubbles_detect)
(CC BY-NC 4.0). An audit found that its official splits weren't safe to use, so this project builds its
own. Details and evidence are in [docs/DATA_NOTES.md](docs/DATA_NOTES.md).

- **Official splits leak.** Every scan source is spread across train, val and test.
- **About half the pages are flipped or rotated copies** of other pages, and some cross splits.
- **Some labels are missing.** In a sample of "empty" pages from one source, about a third had
  visible unlabelled bubbles.

The split here holds out whole sources and never lets a near-duplicate cross a split. Sources with
known label problems stay in train, so val and test are clean.

| Split | Pages | Bubbles |
|---|---|---|
| Train | 1,756 | 10,981 |
| Val | 220 | 1,406 |
| Test | 444 | 4,361 |

## Quick start

Requires Python 3.10+. Commands run from the project root.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.download_data            # About 800 MB
python -m src.prepare_data             # Builds masks and splits
pytest                                 # Runs the test suite
```

### Train

Training needs a GPU. [notebooks/02_train_kaggle.ipynb](notebooks/02_train_kaggle.ipynb) runs the whole
flow on a free Kaggle or Colab GPU.

```bash
python -m src.train --overfit-batch    # Sanity check: the loss must approach 0
python -m src.train                    # Full training (add --resume to continue)
```

### Evaluate

```bash
python -m src.evaluate --checkpoint checkpoints/best.pt --split val --tune   # Tune postprocessing
python -m src.baseline --split val --tune

# Once only, when you're completely done changing things:
python -m src.evaluate --checkpoint checkpoints/best.pt --split test --final
python -m src.baseline --split test --final
```

### Serve

```bash
uvicorn src.api:app --port 8000        # Swagger UI at http://localhost:8000/docs
```

Or with Docker, which mounts the checkpoint so the image contains no weights:

```bash
docker build -t manga-bubble-detector .
docker run -p 8000:8000 -v "$(pwd)/checkpoints:/app/checkpoints:ro" manga-bubble-detector
```

## API

| Endpoint | Method | Returns |
|---|---|---|
| `/health` | GET | Status and model version |
| `/predict` | POST (multipart, field `file`) | `{"boxes": [{x, y, w, h, score}], "inference_ms": ...}` |
| `/predict/image` | POST (multipart, field `file`) | PNG with boxes drawn on it |

```bash
curl -F "file=@page.png" http://localhost:8000/predict
```

Box coordinates refer to the uploaded page in pixels. Uploads must be PNG or JPEG, with the type checked
from the file's contents. Limits are 10 MB (`MAX_UPLOAD_MB`) and 50 megapixels.

## Project layout

```
src/
  download_data.py   Fetch the dataset, pinned to a specific revision
  prepare_data.py    Labels → masks, leak-free splits
  dataset.py         Preprocessing, augmentation, PyTorch Dataset
  model.py           U-Net (nn.Module)
  losses.py          BCE + Dice
  train.py           Training loop, checkpoints, early stopping, resume
  postprocess.py     Mask → boxes
  evaluate.py        Dice, IoU, per-bubble P/R/F1, tuning, error analysis
  baseline.py        OpenCV baseline with no ML
  inference.py       Checkpoint → full page-to-boxes pipeline
  api.py             FastAPI service
configs/default.yaml Hyperparameters
notebooks/           EDA and GPU training
tests/               Unit and end-to-end tests
docs/DATA_NOTES.md   Dataset audit and split design
```

## Limitations

- **The labels are boxes, so the masks are filled rectangles.** The model learns rectangles rather than
  true bubble outlines, and Dice and IoU measure agreement with rectangles. Box metrics are the more
  meaningful numbers here. A polygon-labelled dataset would fix this, and `prepare_data.py` already
  reads that format.
- **The split is source-level, not verified title-level.** The dataset has no title metadata, so image
  resolution stands in for source, backed by visual checks and duplicate detection.
- **Touching bubbles can merge** into one box. Postprocessing has an erosion option that partly helps.
- **Free-floating text and sound effects** can be detected as bubbles, and some labels mark them too.
- **Small bubbles on two-page screenshots** lose detail at 512 px. Metrics are reported per source so
  this stays visible.

## License and credits

Dataset by [PSImera](https://huggingface.co/datasets/PSImera/manga_bubbles_detect), CC BY-NC 4.0. Page
images remain the property of their publishers, so none are included in this repo. Weights trained on
this data are for non-commercial use.
