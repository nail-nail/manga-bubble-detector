import cv2
import numpy as np
import pytest
import torch

from src.dataset import BubbleDataset, augment, collate, letterbox, preprocess_page, scale_boxes
from src.postprocess import Box, boxes_to_original

AUGMENT = {
    "hflip_prob": 0.5,
    "crop_prob": 0.5,
    "crop_scale": [0.6, 1.0],
    "rotate_degrees": 5,
    "brightness": 0.15,
    "contrast": 0.15,
}


def make_record(tmp_path, width=300, height=500, name="page"):
    image = np.full((height, width), 255, dtype=np.uint8)
    mask = np.zeros((height, width), dtype=np.uint8)
    image[100:200, 50:150] = 30
    mask[100:200, 50:150] = 255
    cv2.imwrite(str(tmp_path / f"{name}.png"), image)
    cv2.imwrite(str(tmp_path / f"{name}_mask.png"), mask)
    return {
        "id": name,
        "image": f"{name}.png",
        "mask": f"{name}_mask.png",
        "width": width,
        "height": height,
        "boxes": [[50, 100, 100, 100]],
    }


@pytest.mark.parametrize("width,height", [(300, 500), (500, 300), (512, 512), (1000, 1400), (100, 90)])
def test_letterbox_output_size_and_content_position(width, height):
    image = np.zeros((height, width), dtype=np.uint8)
    canvas, info = letterbox(image, 512)
    assert canvas.shape == (512, 512)
    # The page is black and the padding white, so black pixels show where it landed.
    ys, xs = np.nonzero(canvas == 0)
    assert xs.min() == info.pad_x
    assert ys.min() == info.pad_y
    assert xs.max() - xs.min() + 1 == round(width * info.scale)
    assert ys.max() - ys.min() + 1 == round(height * info.scale)


def test_preprocess_output_is_normalized_to_minus_one_one():
    tensor, _ = preprocess_page(np.array([[0, 255], [255, 0]], dtype=np.uint8), 16)
    assert tensor.shape == (1, 16, 16)
    assert tensor.min() >= -1.0 and tensor.max() <= 1.0


def test_box_round_trip_from_page_to_network_and_back():
    _, info = letterbox(np.zeros((1400, 1000), dtype=np.uint8), 512)
    original = np.array([[120, 340, 200, 180]], dtype=np.float32)
    x, y, w, h = scale_boxes(original, info)[0]
    box = Box(round(float(x)), round(float(y)), round(float(w)), round(float(h)), 1.0)
    back = boxes_to_original([box], info.scale, info.pad_x, info.pad_y, info.orig_w, info.orig_h)[0]
    assert abs(back.x - 120) <= 3 and abs(back.y - 340) <= 3
    assert abs(back.w - 200) <= 3 and abs(back.h - 180) <= 3


def test_validation_item_has_aligned_image_mask_and_boxes(tmp_path):
    record = make_record(tmp_path)
    item = BubbleDataset([record], 512, root=tmp_path)[0]
    assert item["image"].shape == (1, 512, 512)
    assert item["mask"].shape == (1, 512, 512)
    assert set(item["mask"].unique().tolist()) == {0.0, 1.0}

    # The mask must cover the dark square in the image, and the box must enclose the mask.
    mask = item["mask"][0].numpy().astype(bool)
    assert item["image"][0].numpy()[mask].mean() < -0.5
    x, y, w, h = item["boxes"][0].tolist()
    ys, xs = np.nonzero(mask)
    assert abs(xs.min() - x) <= 1 and abs(ys.min() - y) <= 1
    assert abs(xs.max() + 1 - (x + w)) <= 1 and abs(ys.max() + 1 - (y + h)) <= 1


def test_augmented_mask_stays_binary_and_aligned_with_the_image():
    image = np.full((400, 300), 255, dtype=np.uint8)
    mask = np.zeros((400, 300), dtype=np.uint8)
    image[100:250, 80:220] = 20
    mask[100:250, 80:220] = 255
    cfg = {**AUGMENT, "brightness": 0.0, "contrast": 0.0}
    for seed in range(30):
        out_image, out_mask = augment(image, mask, cfg, np.random.default_rng(seed))
        assert out_image.shape == out_mask.shape
        assert set(np.unique(out_mask).tolist()) <= {0, 255}
        if (out_mask == 255).sum() > 500:
            # Most masked pixels must still sit on the dark square after flip/rotate/crop.
            assert (out_image[out_mask == 255] < 128).mean() > 0.9


def test_each_epoch_produces_a_different_augmentation(tmp_path):
    record = make_record(tmp_path)
    dataset = BubbleDataset([record], 64, augment_cfg=AUGMENT, root=tmp_path)
    torch.manual_seed(0)
    images = [dataset[0]["image"] for _ in range(4)]
    assert any(not torch.equal(images[0], other) for other in images[1:])


def test_global_seed_reproduces_augmentation(tmp_path):
    record = make_record(tmp_path)
    dataset = BubbleDataset([record], 64, augment_cfg=AUGMENT, root=tmp_path)
    torch.manual_seed(7)
    first = dataset[0]["image"]
    torch.manual_seed(7)
    second = dataset[0]["image"]
    assert torch.equal(first, second)


def test_collate_keeps_variable_length_boxes_as_a_list(tmp_path):
    first = make_record(tmp_path, name="a")
    second = {**make_record(tmp_path, name="b"), "boxes": []}
    dataset = BubbleDataset([first, second], 64, root=tmp_path)
    batch = collate([dataset[0], dataset[1]])
    assert batch["image"].shape == (2, 1, 64, 64)
    assert [len(b) for b in batch["boxes"]] == [1, 0]
    assert batch["id"] == ["a", "b"]
