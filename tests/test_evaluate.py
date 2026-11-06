import numpy as np
import pytest

from src.baseline import BaselineParams, detect
from src.evaluate import (
    PageResult,
    box_iou,
    f1_from_counts,
    match_boxes,
    pixel_scores,
    summarize,
    summarize_by_source,
    tune_postprocess,
)
from src.postprocess import Box


def box(x, y, w, h, score=0.9):
    return Box(x, y, w, h, score)


def test_pixel_scores_known_values():
    pred = np.zeros((10, 10), dtype=bool)
    target = np.zeros((10, 10), dtype=bool)
    pred[:, :5] = True  # 50 px
    target[:, 2:7] = True  # 50 px, 30 of them shared
    dice, iou = pixel_scores(pred, target)
    assert dice == pytest.approx(0.6)
    assert iou == pytest.approx(30 / 70)


def test_pixel_scores_two_empty_masks_agree_perfectly():
    empty = np.zeros((4, 4), dtype=bool)
    assert pixel_scores(empty, empty) == (1.0, 1.0)


def test_pixel_scores_prediction_on_an_empty_page_scores_zero():
    empty = np.zeros((4, 4), dtype=bool)
    assert pixel_scores(np.ones((4, 4), dtype=bool), empty) == (0.0, 0.0)


def test_box_iou_known_values():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert box_iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)
    assert box_iou((0, 0, 10, 10), (10, 0, 10, 10)) == 0.0  # Touching edges share no area.


def test_match_counts_hits_misses_and_false_alarms():
    preds = [box(0, 0, 10, 10), box(100, 100, 10, 10)]
    targets = [(0, 0, 10, 10), (50, 50, 10, 10)]
    assert match_boxes(preds, targets) == (1, 1, 1)


def test_match_is_one_to_one_so_a_double_detection_is_a_false_positive():
    preds = [box(0, 0, 10, 10, 0.9), box(0, 0, 10, 10, 0.8)]
    assert match_boxes(preds, [(0, 0, 10, 10)]) == (1, 1, 0)


def test_match_respects_the_iou_threshold_boundary():
    target = [(0, 0, 10, 10)]
    assert match_boxes([box(0, 0, 10, 5)], target) == (1, 0, 0)  # IoU exactly 0.5
    assert match_boxes([box(0, 0, 10, 4)], target) == (0, 1, 1)  # IoU 0.4


def test_match_gives_the_contested_target_to_the_higher_score():
    target = [(0, 0, 10, 10)]
    weak_but_tight = box(0, 0, 10, 10, score=0.5)
    strong_but_loose = box(0, 0, 10, 6, score=0.9)
    assert match_boxes([weak_but_tight, strong_but_loose], target) == (1, 1, 0)


def test_match_handles_empty_inputs():
    assert match_boxes([], []) == (0, 0, 0)
    assert match_boxes([box(0, 0, 5, 5)], []) == (0, 1, 0)
    assert match_boxes([], [(0, 0, 5, 5)]) == (0, 0, 1)


def test_f1_from_counts():
    assert f1_from_counts(8, 2, 8) == pytest.approx((0.8, 0.5, 2 * 0.8 * 0.5 / 1.3))
    assert f1_from_counts(0, 0, 0) == (0.0, 0.0, 0.0)


def test_page_f1_is_perfect_for_a_correctly_empty_page():
    assert PageResult("a", 1.0, 1.0, 0, 0, 0).f1 == 1.0
    assert PageResult("a", 0.0, 0.0, 0, 3, 0).f1 == 0.0


def test_summarize_pools_bubbles_but_averages_dice_per_page():
    results = [PageResult("a", 1.0, 1.0, 9, 0, 0), PageResult("b", 0.0, 0.0, 0, 1, 1)]
    summary = summarize(results)
    assert summary["dice"] == pytest.approx(0.5)
    assert summary["precision"] == pytest.approx(0.9)
    assert summary["recall"] == pytest.approx(0.9)


def test_summarize_by_source_separates_sources():
    results = [PageResult("a", 1.0, 1.0, 1, 0, 0), PageResult("b", 0.0, 0.0, 0, 1, 0)]
    records = [{"id": "a", "source": "s1"}, {"id": "b", "source": "s2"}]
    by_source = summarize_by_source(results, records)
    assert by_source["s1"]["dice"] == 1.0
    assert by_source["s2"]["dice"] == 0.0


def test_tuning_picks_values_that_remove_a_noise_blob():
    prob = np.zeros((128, 128), dtype=np.float32)
    prob[20:60, 20:60] = 0.9  # A real bubble.
    prob[100:108, 100:108] = 0.9  # A noise blob of 64 px.
    sample = {
        "id": "a",
        "prob": prob,
        "mask": prob > 0.5,
        "boxes": np.array([[20, 20, 40, 40]], dtype=np.float32),
    }
    grid = {"threshold": [0.5], "open_kernel": [3], "min_area": [10, 200], "erode_px": [0]}
    params, summary = tune_postprocess([sample], grid)
    assert params.min_area == 200
    assert summary["f1"] == 1.0


def test_baseline_finds_a_white_bubble_with_text_on_a_dark_page():
    page = np.full((256, 256), 60, dtype=np.uint8)
    page[50:110, 60:140] = 255
    page[70:80, 80:120] = 0  # "Text" inside the bubble.
    mask, boxes = detect(page, BaselineParams())
    assert len(boxes) == 1
    assert (boxes[0].x, boxes[0].y, boxes[0].w, boxes[0].h) == (60, 50, 80, 60)
    assert mask[75, 100]  # Holes made by text are filled.


def test_baseline_ignores_empty_white_regions_and_page_margins():
    page = np.full((256, 256), 60, dtype=np.uint8)
    page[50:110, 60:140] = 255  # Right size and shape, but with no text inside.
    page[:, :20] = 255  # A white page margin touching the edge.
    _, boxes = detect(page, BaselineParams(min_ink_frac=0.01))
    assert boxes == []
