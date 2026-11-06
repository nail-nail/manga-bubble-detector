import numpy as np
import pytest

from src.postprocess import Box, PostprocessParams, boxes_to_original, mask_to_boxes


def blank(size: int = 128) -> np.ndarray:
    return np.zeros((size, size), dtype=np.float32)


def test_empty_mask_gives_no_boxes():
    assert mask_to_boxes(blank()) == []


def test_single_blob_gives_exact_box_and_mean_score():
    prob = blank()
    prob[10:30, 20:60] = 0.8  # 20 tall, 40 wide
    boxes = mask_to_boxes(prob, PostprocessParams(open_kernel=0, min_area=10))
    assert len(boxes) == 1
    box = boxes[0]
    assert (box.x, box.y, box.w, box.h) == (20, 10, 40, 20)
    assert box.score == pytest.approx(0.8)


def test_two_separate_blobs_are_two_boxes_sorted_by_score():
    prob = blank()
    prob[5:25, 5:25] = 0.6
    prob[60:90, 60:90] = 0.9
    boxes = mask_to_boxes(prob, PostprocessParams(open_kernel=0, min_area=10))
    assert [b.score for b in boxes] == [pytest.approx(0.9), pytest.approx(0.6)]


def test_small_components_are_removed_by_min_area():
    prob = blank()
    prob[10:13, 10:13] = 0.9  # area 9
    prob[50:80, 50:80] = 0.9  # area 900
    boxes = mask_to_boxes(prob, PostprocessParams(open_kernel=0, min_area=50))
    assert len(boxes) == 1
    assert boxes[0].x == 50


def test_threshold_is_strict_and_configurable():
    prob = blank()
    prob[10:40, 10:40] = 0.5
    assert mask_to_boxes(prob, PostprocessParams(threshold=0.5, open_kernel=0, min_area=1)) == []
    assert len(mask_to_boxes(prob, PostprocessParams(threshold=0.4, open_kernel=0, min_area=1))) == 1


def test_opening_removes_a_one_pixel_speck():
    prob = blank()
    prob[50, 50] = 1.0
    params = PostprocessParams(open_kernel=3, min_area=1)
    assert mask_to_boxes(prob, params) == []


def test_erosion_splits_a_thin_bridge_and_restores_box_size():
    prob = blank()
    prob[20:60, 10:50] = 0.9
    prob[20:60, 70:110] = 0.9
    prob[38:42, 50:70] = 0.9  # 4 px bridge joining the two blobs

    merged = mask_to_boxes(prob, PostprocessParams(open_kernel=0, min_area=10))
    assert len(merged) == 1

    split = mask_to_boxes(prob, PostprocessParams(open_kernel=0, min_area=10, erode_px=3))
    assert len(split) == 2
    left = min(split, key=lambda b: b.x)
    assert (left.x, left.y, left.h) == (10, 20, 40)
    # The bridge's root survives erosion as a small nub, so allow 1 px on that side.
    assert left.w in (40, 41)


def test_box_is_clipped_to_image_when_grown_after_erosion():
    prob = blank(64)
    prob[0:30, 0:30] = 0.9
    box = mask_to_boxes(prob, PostprocessParams(open_kernel=0, min_area=10, erode_px=3))[0]
    assert box.x >= 0 and box.y >= 0
    assert box.x + box.w <= 64 and box.y + box.h <= 64


def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        mask_to_boxes(np.zeros((1, 64, 64)))


def test_boxes_to_original_undoes_letterbox():
    # A 1000x500 page in a 512 canvas: scale 0.512, 128 px of padding top and bottom.
    box = Box(x=51, y=128 + 25, w=256, h=128, score=0.9)
    out = boxes_to_original([box], scale=0.512, pad_x=0, pad_y=128, orig_w=1000, orig_h=500)[0]
    assert (out.x, out.y) == (100, 49)
    assert out.w == 500
    assert out.h == 250
    assert out.score == 0.9


def test_boxes_to_original_clips_and_drops_boxes_in_padding():
    inside_padding = Box(x=10, y=0, w=20, h=20, score=0.5)
    straddling = Box(x=10, y=120, w=20, h=20, score=0.5)
    out = boxes_to_original(
        [inside_padding, straddling], scale=0.5, pad_x=0, pad_y=128, orig_w=1024, orig_h=512
    )
    assert len(out) == 1
    assert out[0].y == 0
