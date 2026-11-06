import json

import pytest

from src.evaluate import assign_boxes, match_boxes
from src.evaluate_external import load_labelstudio, score_pages, to_pixel_box
from src.postprocess import Box


def region(label, x, y, w, h, width=200, height=400):
    """A Label Studio rectangle. Its x, y, w, h are percentages of the image size."""
    return {
        "type": "rectanglelabels", "original_width": width, "original_height": height,
        "value": {"x": x, "y": y, "width": w, "height": h, "rotation": 0, "rectanglelabels": [label]},
    }


def write_export(tmp_path, tasks):
    path = tmp_path / "export.json"
    path.write_text(json.dumps(tasks), encoding="utf-8")
    return path


def test_load_labelstudio_converts_percentages_to_pixels(tmp_path):
    export = write_export(tmp_path, [{
        "file_upload": "81af392a-op_001.jpg", "data": {"image": "/data/upload/1/81af392a-op_001.jpg"},
        "annotations": [{"was_cancelled": False, "result": [region("speech", 10, 25, 50, 10)]}],
    }])
    [page] = load_labelstudio(export, tmp_path / "images")
    assert page["id"] == "op_001"
    assert page["image"].endswith("op_001.jpg")
    assert page["boxes"] == [pytest.approx([20, 100, 100, 40])]
    assert page["labels"] == ["speech"]


def test_to_pixel_box_turns_a_rotated_box_into_its_upright_envelope():
    # 40 x 20 px at (100, 100), turned 90° clockwise about its top-left corner.
    value = {"x": 50, "y": 25, "width": 20, "height": 5, "rotation": 90}
    assert to_pixel_box(value, 200, 400) == pytest.approx([80, 100, 20, 40])
    value["rotation"] = 0
    assert to_pixel_box(value, 200, 400) == pytest.approx([100, 100, 40, 20])


def test_load_labelstudio_skips_unlabelled_and_cancelled_tasks(tmp_path):
    export = write_export(tmp_path, [
        {"file_upload": "a-op_001.jpg", "data": {}, "annotations": []},
        {"file_upload": "b-op_002.jpg", "data": {}, "annotations": [{"was_cancelled": True, "result": []}]},
    ])
    assert load_labelstudio(export, tmp_path) == []


def test_load_labelstudio_rejects_unknown_classes(tmp_path):
    export = write_export(tmp_path, [{
        "file_upload": "a-op_001.jpg", "data": {},
        "annotations": [{"was_cancelled": False, "result": [region("bubble", 0, 0, 10, 10)]}],
    }])
    with pytest.raises(ValueError, match="unknown class"):
        load_labelstudio(export, tmp_path)


def test_assign_boxes_matches_match_boxes():
    preds = [Box(0, 0, 10, 10, 0.9), Box(100, 100, 10, 10, 0.8), Box(1, 1, 10, 10, 0.7)]
    targets = [[0, 0, 10, 10], [50, 50, 10, 10]]
    assert assign_boxes(preds, targets) == {0: 0}
    assert match_boxes(preds, targets) == (1, 2, 1)


def test_score_pages_treats_other_classes_as_neutral():
    page = {
        "id": "p", "boxes": [[0, 0, 10, 10], [20, 0, 10, 10], [40, 0, 10, 10], [60, 0, 10, 10]],
        "labels": ["speech", "thought", "sfx", "narration"],
    }
    preds = [
        Box(0, 0, 10, 10, 0.9),     # speech: a hit
        Box(40, 0, 10, 10, 0.8),    # sfx: neutral, not a false positive
        Box(100, 100, 10, 10, 0.7), # matches nothing: a false positive
    ]
    summary = score_pages([page], {"p": preds})
    bubbles = summary["bubbles"]
    assert (bubbles["tp"], bubbles["fp"], bubbles["fn"]) == (1, 1, 1)  # the thought balloon is missed
    assert summary["neutral_matches"] == 1
    assert summary["recall_by_class"]["sfx"] == {"found": 1, "total": 1, "recall": 1.0}
    assert summary["recall_by_class"]["narration"]["recall"] == 0.0
    assert summary["recall_by_class"]["other"]["recall"] is None
