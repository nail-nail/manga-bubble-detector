# Labelling guide: external test set

Rules for hand-labelling the external test pages in Label Studio
([configs/label_studio.xml](../configs/label_studio.xml)). The goal is to measure how well the detector
finds speech bubbles on a series it never saw, and which kinds of text it misses.

Label from scratch. Don't look at model predictions before or while labelling: seeing them pulls the
labels towards the model and inflates the score.

## Classes

Every text region on the page gets one box and exactly one class.

| Key | Class | What it covers |
|---|---|---|
| 1 | `speech` | A balloon with a drawn outline that holds spoken words. Includes shouting balloons with spiky or jagged outlines. |
| 2 | `thought` | A thought balloon: cloud-shaped outline, or a tail made of small circles. |
| 3 | `narration` | A caption box, usually rectangular, with narrative text. |
| 4 | `sfx` | A sound effect drawn into the art, outside any balloon (ドン, ゴゴゴ, BOOM). |
| 5 | `other` | Any other text: words with no balloon outline, signs, chapter titles, panel numbers, character name and bounty tags, author notes. |

If you can't decide between two classes, pick `other` and move on.

Two cases that came up in the first batch:

- **Decide `speech` vs `narration` by who is talking, not by the box shape.** A character speaking is
  `speech` even inside a rectangular or octagonal box. In One Piece, octagonal boxes carry voices over
  a Den Den Mushi or a loudspeaker. `narration` is the narrator's voice: time, place, background.
- **Character name and bounty tags are `other`,** even though they sit in a box. So are title cards
  with a character's name.
- **Flashback balloons are `speech`.** One Piece draws remembered dialogue in balloons with a fuzzy,
  radiating outline. They aren't thoughts.

## Box extent

- **`speech`, `thought`, `narration`:** draw the box around the balloon or box **outline**, not just
  the text. Leave out the balloon's tail.
- **`sfx`, `other`:** draw the box around the **text** itself.
- **Joined balloons:** if two balloons share an outline but each lobe holds its own block of text,
  give each lobe its own box. If it's one balloon with one block of text, use one box.
- **Balloons cut by a panel edge or the page edge:** box only the visible part.
- **Furigana** (small reading hints beside kanji) belong to the text they annotate, so they're inside
  the same box.

### What happened on the One Piece set

Most `speech` boxes were first drawn around the text rather than the balloon outline. Since the
training labels and the detector box the whole balloon, those boxes would have failed the IoU ≥ 0.5
match even on a correct prediction. [src/snap_labels.py](../src/snap_labels.py) grew each `speech` and
`thought` box out to the outline around it, using image processing only, never model predictions:
537 of 572 boxes were snapped and every page was checked by eye. The other 35 (no closed outline,
or a leak into the art) were kept as drawn and checked by hand in Label Studio.

## What to skip

- Text that is too small or blurry to read at 100% zoom.
- Pages or regions so warped by the photo that the outline can't be traced. Note the page ID instead.

## How it's scored

The detector predicts one class, "bubble", so scoring uses the classes as subsets:

- **Main score:** precision, recall and F1 against `speech` + `thought` boxes, at box IoU ≥ 0.5.
- **Recall per class:** how many boxes of each class the detector finds. `sfx` and `other` show what
  the training labels did or didn't teach the model.
- **False positives:** a predicted box counts as a false positive only if it matches no labelled box
  of any class.
