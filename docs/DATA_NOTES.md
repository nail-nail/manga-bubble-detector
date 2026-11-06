# Data notes

What was found when auditing the dataset, and what was done about it. These findings are why this
project doesn't use the dataset's own train/val/test splits.

## Dataset choice

| Candidate | Annotations | Pages | License | Outcome |
|---|---|---|---|---|
| [`PSImera/manga_bubbles_detect`](https://huggingface.co/datasets/PSImera/manga_bubbles_detect) | Boxes only | 2,671 | CC BY-NC 4.0 | **Used** |
| [`pers-kdqu3/manga-2-segmentation`](https://universe.roboflow.com/pers-kdqu3/manga-2-segmentation) | Instance segmentation (per search listing) | About 2,810 | Not confirmed | Worth checking: it would remove the main limitation below |
| [`wouter-bfixj/mangaspeechbubbles`](https://universe.roboflow.com/wouter-bfixj/mangaspeechbubbles) | Instance segmentation | About 30 | CC BY 4.0 | Too few pages |
| [`karimbasem42/manga-speech-bubble-detection`](https://universe.roboflow.com/karimbasem42-gmail-com/manga-speech-bubble-detection) | Not confirmed | Not confirmed | Not confirmed | Page could not be read |
| Manga109 | Boxes | - | Academic request | Backup only: access takes time |

The Roboflow details come from search listings, because those pages block automated access. Confirm
them on the live pages before relying on them.

`src/prepare_data.py` reads both YOLO boxes and YOLO polygons, so a mask dataset can be swapped in
without changing the rest of the pipeline.

## Findings

### 1. The official splits leak by source

Grouping pages by image resolution shows every large scan source spread across all three official
splits. For example, the 844 x 1200 family is split 295 / 72 / 40. Test pages therefore share an art
style, and often a series, with training pages, which is the leak this project set out to avoid.

### 2. About half the pages are flipped or rotated copies

A perceptual-hash comparison across six orientations (original, mirrored, flipped, and rotated by
90, 180 and 270 degrees) put 1,297 of 2,671 pages in a near-duplicate group at Hamming distance 10.
Most groups are pairs. These look like augmented copies from an export step, which also explains why
some pages are upside down. At least 12 of these groups cross an official split boundary.

The check needed all orientations: an upright-only comparison found just 129 near-duplicate pairs.

### 3. Empty labels aren't reliable in the 640 x 640 source

322 pages have no boxes, and 245 of them are in the 640 x 640 source. In a random sample of 30 of
those, roughly a third had visible speech bubbles or text boxes with no labels (for example
`train/img_001374`), and the rest were genuinely empty, such as covers and full-page art. Some
non-empty pages in that source are also only partly labelled. A sample of 36 empty pages from other
sources looked genuinely free of bubbles at contact-sheet resolution.

Dropping all 245 also loses real negatives, but 40 empty pages from other sources remain in train,
and the many background pixels on labelled pages still teach the model what isn't a bubble.

### 4. The 640 x 640 pages are distorted

They were resized to a square from other aspect ratios, which squashes the bubbles. Other sources
keep their original proportions.

### 5. 3840 x 2160 pages are two-page screenshots

They have dark bars on both sides, and bubbles become very small after downscaling to 512 px. Expect
lower scores on this source, which is why evaluation reports metrics per source.

## How the split is built

There is no title metadata and the filenames are anonymized. Neighbouring file numbers look no more
alike than random pairs, so the numbering says nothing about chapters either. Following the fallback
of grouping by source ID, image resolution is used as the source ID.

| Split | Sources | Pages | Share | Bubbles |
|---|---|---|---|---|
| Train | `square_640`, `portrait_844`, `other` | 1,756 | 72.6% | 10,981 |
| Val | `portrait_2150` | 220 | 9.1% | 1,406 |
| Test | `portrait_764`, `screens_3840`, `scan_1044` | 444 | 18.3% | 4,361 |

The rules, in order:

1. Drop the 245 empty-label pages from `square_640`. Keep empty pages from other sources, so the model
   still gets evaluated on pages with no bubbles.
2. Assign whole sources to a split, keeping all sources with known label problems in train. Val and
   test contain only clean, undistorted pages.
3. If a near-duplicate group crosses splits, keep it in the most protected split and drop the other
   copies. This removed 6 pages.

Page IDs keep the dataset's original folder name, such as `test_img_000151`, so they can be traced back
to the raw files. That prefix is the **original** split, not this project's: a page called `test_...`
can sit in this project's val set.

The ratio is 73/9/18 rather than the planned 70/15/15, because whole sources can't be divided to hit
an exact ratio. Test was given more sources so its score reflects more than one series.

## Limitations

- **Sources are inferred, not verified.** Resolution stands in for title. Contact sheets suggest each
  held-out source is one or a few series not visible in the training pool, and those sources share no
  near-duplicates with `square_640`. That sample-based check can't rule out overlap completely, so
  this is best described as a **source-level** split rather than a title-level one.
- **Masks are filled rectangles.** The labels are boxes, so the model learns to predict rectangles,
  and Dice and IoU measure agreement with rectangles rather than true bubble outlines. Box metrics at
  IoU 0.5 are the more meaningful numbers here.
- **Touching boxes merge.** Overlapping filled rectangles form one blob, so instance recall suffers
  on crowded pages. The `erode_px` postprocessing option partly addresses this.
- **Training labels are noisy.** Partly labelled pages and upside-down copies remain in train, because
  the copies can't be told apart from their originals automatically. Noise in train can hurt the
  score but can't inflate it, since val and test don't contain these pages.
- **Not every box is a bubble.** Some boxes mark free-floating text outside any bubble.

## Copyright

The labels are CC BY-NC 4.0, and the images remain the property of their publishers. So this repo
contains no dataset images, only a download script and page IDs. Generated overlays are gitignored.
Trained weights come from NC-licensed labels, so publish them for non-commercial use only and credit
the dataset.
