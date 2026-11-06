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

Test set: 444 pages and 4,361 bubbles from sources the model never trained on. It was evaluated once,
after all choices were final. Precision, recall and F1 pool all bubbles, with a match requiring box
IoU ≥ 0.5.

| Model | Dice | IoU | Precision @0.5 | Recall @0.5 | F1 | Params | CPU ms/page |
|---|---|---|---|---|---|---|---|
| Baseline (OpenCV, no ML) | 0.444 | 0.326 | 0.823 | 0.332 | 0.473 | 0 | 6 |
| **U-Net (from scratch)** | **0.802** | **0.713** | **0.886** | **0.607** | **0.720** | 7.8M | 1,228 |

CPU latency is the median per page for preprocessing, the U-Net at 512 px and postprocessing, not
counting image decoding, on an Intel Core Ultra 9 185H. Training took 41 epochs on a Kaggle T4, about 85 s each, with early stopping keeping
epoch 33.

### By test source

| Source | Pages | Bubbles | Precision | Recall | F1 | Baseline F1 |
|---|---|---|---|---|---|---|
| `portrait_764` | 188 | 1,545 | 0.867 | 0.640 | 0.736 | 0.432 |
| `scan_1044` | 164 | 1,601 | 0.930 | 0.587 | 0.719 | 0.547 |
| `screens_3840` | 92 | 1,215 | 0.858 | 0.591 | 0.700 | 0.415 |

### Validation vs test

| Split | Precision | Recall | F1 |
|---|---|---|---|
| Val (`portrait_2150`, used for tuning) | 0.907 | 0.772 | 0.834 |
| Test (3 unseen sources) | 0.886 | 0.607 | 0.720 |

Precision holds up on new sources. Recall drops, so the model mostly fails by **missing** labels rather
than inventing them. The postprocessing threshold (0.9) was tuned on a single validation source, and a
model is usually less confident on art it hasn't seen, so a lower threshold might suit new sources
better. That can't be checked without a second validation source, because tuning on test would leak
it.

### Error analysis

The 10 worst pages of each split, ranked by per-page F1, are mostly labels for text that isn't in a
speech bubble:

- **Validation:** chapter titles, a contents page, borderless captions and handwritten text. The label
  boxes the whole text block while the model boxes a fragment, or erosion splits it. The rest are false
  detections on sound effects and a shop sign on pages with no bubbles.
- **Test:** panel numbers, character name tags, a Korean chapter title and sound-effect lettering.
  The model predicts nothing on these pages, which is arguably right for a *bubble* detector, but each
  one counts as a miss.

So part of the recall gap comes from the dataset's broad idea of a "bubble". Ranking by per-page F1
favours pages with only a few labels, though, so this list doesn't show where most of the 1,715 test
misses are. Touching bubbles merging into one box are a likely source on busy pages.

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

- **Code:** MIT, see [LICENSE](LICENSE).
- **Dataset:** by [PSImera](https://huggingface.co/datasets/PSImera/manga_bubbles_detect), CC BY-NC 4.0.
  Page images remain the property of their publishers, so none are included in this repo.
- **Trained weights:** derived from that dataset, so they are for non-commercial use only.
- **Demo art:** [Pepper&Carrot](https://www.peppercarrot.com) by David Revoy, CC BY 4.0.
