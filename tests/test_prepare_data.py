import numpy as np
import pytest

from src.prepare_data import (
    duplicate_groups,
    parse_label_line,
    rasterize,
    resolve_duplicate_conflicts,
    source_of,
)


def test_parse_detection_box_line():
    kind, points = parse_label_line("0 0.5 0.5 0.2 0.4", 100, 200)
    assert kind == "box"
    assert points.min(axis=0).tolist() == pytest.approx([40, 60])
    assert points.max(axis=0).tolist() == pytest.approx([60, 140])


def test_parse_polygon_line():
    kind, points = parse_label_line("0 0.1 0.1 0.9 0.1 0.5 0.9", 100, 100)
    assert kind == "polygon"
    assert points.shape == (3, 2)


def test_blank_line_is_skipped_and_malformed_line_raises():
    assert parse_label_line("   ", 10, 10) is None
    with pytest.raises(ValueError):
        parse_label_line("0 0.1 0.2 0.3", 10, 10)


def test_rasterize_box_fills_mask_and_records_box():
    mask, boxes, kind = rasterize(["0 0.5 0.5 0.5 0.5"], 100, 100)
    assert kind == "box"
    assert boxes == [[25, 25, 50, 50]]
    assert mask[50, 50] == 255
    assert mask[5, 5] == 0


def test_rasterize_polygon_follows_the_shape_rather_than_its_box():
    mask, boxes, kind = rasterize(["0 0.1 0.1 0.9 0.1 0.1 0.9"], 100, 100)
    assert kind == "polygon"
    assert boxes == [[10, 10, 80, 80]]
    assert mask[20, 20] == 255
    assert mask[85, 85] == 0  # Inside the box, but outside the triangle.


def test_rasterize_keeps_touching_boxes_as_separate_instances():
    _, boxes, _ = rasterize(["0 0.25 0.5 0.5 0.5", "0 0.75 0.5 0.5 0.5"], 100, 100)
    assert len(boxes) == 2


def test_rasterize_clips_boxes_that_leave_the_image_and_drops_degenerate_ones():
    _, boxes, _ = rasterize(["0 0.95 0.5 0.2 0.2", "0 0.5 0.5 0 0"], 100, 100)
    assert boxes == [[85, 40, 15, 20]]


@pytest.mark.parametrize(
    "size,expected",
    [
        ((640, 640), "square_640"),
        ((3840, 2160), "screens_3840"),
        ((764, 1200), "portrait_764"),
        ((844, 1200), "portrait_844"),
        ((1447, 2150), "portrait_2150"),
        ((1440, 1044), "scan_1044"),
        ((999, 777), "other"),
    ],
)
def test_source_of(size, expected):
    assert source_of(*size) == expected


def fake_hashes(*upright_values):
    """Pages whose six orientations share the same hash, for testing the grouping logic."""
    hashes = np.zeros((len(upright_values), 6, 64), dtype=bool)
    for index, flipped_bits in enumerate(upright_values):
        hashes[index, :, :flipped_bits] = True
    return hashes


def test_near_duplicates_group_together_and_distinct_pages_do_not():
    groups = duplicate_groups(fake_hashes(0, 3, 40), max_distance=10)
    assert groups[0] == groups[1]
    assert groups[2] != groups[0]


def test_duplicate_detection_chains_through_an_intermediate_page():
    groups = duplicate_groups(fake_hashes(0, 8, 16), max_distance=10)
    assert groups[0] == groups[1] == groups[2]


def test_flipped_copy_is_found_through_another_orientation():
    hashes = np.zeros((2, 6, 64), dtype=bool)
    hashes[0, 0, :32] = True  # Page A upright differs a lot from B upright...
    hashes[1, 0, 32:] = True
    hashes[1, 2, :32] = True  # ...but B flipped vertically matches A.
    groups = duplicate_groups(hashes, max_distance=4)
    assert groups[0] == groups[1]


def test_conflicts_keep_the_most_protected_split():
    splits = ["train", "test", "val", "train", "val"]
    groups = [1, 1, 2, 2, 3]
    assert resolve_duplicate_conflicts(splits, groups) == [False, True, True, False, True]


def test_no_group_spans_two_splits_after_resolution():
    rng = np.random.default_rng(0)
    splits = rng.choice(["train", "val", "test"], size=500).tolist()
    groups = rng.integers(0, 60, size=500).tolist()
    keep = resolve_duplicate_conflicts(splits, groups)
    seen: dict[int, str] = {}
    for split, group, flag in zip(splits, groups, keep):
        if flag:
            assert seen.setdefault(group, split) == split
