"""End-to-end training tests on a tiny synthetic dataset. They run on CPU in seconds."""

import copy
import csv
import json

import cv2
import numpy as np
import pytest
import torch

from src.config import load_config
from src.inference import Predictor
from src.train import build_parser, fit, overfit_one_batch


@pytest.fixture
def project(tmp_path, monkeypatch):
    """A temporary project with fake pages: dark squares on white paper."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "pages").mkdir()
    (tmp_path / "splits").mkdir()
    rng = np.random.default_rng(0)
    records = []
    for index in range(8):
        image = np.full((64, 64), 255, dtype=np.uint8)
        mask = np.zeros((64, 64), dtype=np.uint8)
        x, y = rng.integers(4, 30, size=2).tolist()
        image[y : y + 24, x : x + 24] = 40
        mask[y : y + 24, x : x + 24] = 255
        cv2.imwrite(f"pages/{index}.png", image)
        cv2.imwrite(f"pages/{index}_mask.png", mask)
        records.append(
            {"id": f"p{index}", "image": f"pages/{index}.png", "mask": f"pages/{index}_mask.png",
             "width": 64, "height": 64, "boxes": [[x, y, 24, 24]], "source": "fake"}
        )
    (tmp_path / "splits" / "train.json").write_text(json.dumps(records[:6]))
    (tmp_path / "splits" / "val.json").write_text(json.dumps(records[6:]))
    return tmp_path


CONFIG = {
    "seed": 1,
    "data": {"processed_dir": "processed", "splits_dir": "splits", "image_size": 32, "num_workers": 0},
    "model": {"in_channels": 1, "base_channels": 4},
    "loss": {"bce_weight": 1.0, "dice_weight": 1.0},
    "train": {
        "epochs": 2, "batch_size": 2, "lr": 0.01, "weight_decay": 0.0001,
        "early_stopping_patience": 8, "mixed_precision": True,
        "checkpoint_dir": "ckpt", "log_csv": "logs/log.csv",
    },
    "augment": {
        "hflip_prob": 0.5, "crop_prob": 0.5, "crop_scale": [0.6, 1.0],
        "rotate_degrees": 5, "brightness": 0.15, "contrast": 0.15,
    },
    "postprocess": {"threshold": 0.5, "open_kernel": 3, "min_area": 4, "erode_px": 0},
}


def make_config(**train_overrides):
    config = copy.deepcopy(CONFIG)
    config["train"].update(train_overrides)
    return config


def read_log():
    with open("logs/log.csv", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def test_overfit_one_batch_drives_the_loss_down(project):
    torch.manual_seed(0)
    losses = overfit_one_batch(make_config(batch_size=4), torch.device("cpu"), steps=60)
    assert losses[-1] < 0.25 * losses[0]


def test_fit_writes_checkpoints_and_a_log(project):
    result = fit(make_config(), torch.device("cpu"))
    assert result["best_dice"] >= 0.0
    assert (project / "ckpt" / "best.pt").exists()
    assert (project / "ckpt" / "last.pt").exists()
    assert [row["epoch"] for row in read_log()] == ["1", "2"]


def test_best_checkpoint_loads_safely_and_can_predict(project):
    fit(make_config(), torch.device("cpu"))
    predictor = Predictor(project / "ckpt" / "best.pt")
    boxes, elapsed_ms = predictor.predict(np.full((64, 64), 255, dtype=np.uint8))
    assert isinstance(boxes, list)
    assert elapsed_ms > 0


def test_resume_continues_from_the_next_epoch_and_keeps_the_log(project):
    fit(make_config(epochs=2), torch.device("cpu"))
    lr_before = torch.load(project / "ckpt" / "last.pt", weights_only=True)["scheduler"]["last_epoch"]
    assert lr_before == 2
    fit(make_config(epochs=4), torch.device("cpu"), resume=True)
    state = torch.load(project / "ckpt" / "last.pt", weights_only=True)
    assert state["epoch"] == 4
    assert [row["epoch"] for row in read_log()] == ["1", "2", "3", "4"]


def test_resume_with_more_epochs_follows_the_new_cosine_curve(project):
    # Regression test: CosineAnnealingLR kept its old cycle length after resuming,
    # so extending from 2 to 4 epochs trained epoch 3 at an LR of 0.
    fit(make_config(epochs=2, lr=0.01), torch.device("cpu"))
    fit(make_config(epochs=4, lr=0.01), torch.device("cpu"), resume=True)
    logged = [float(row["lr"]) for row in read_log()]
    expected = [0.01 * 0.5 * (1 + np.cos(np.pi * e / 4)) for e in range(4)]
    # Epochs 1-2 ran under the original 2-epoch curve. From the resume on, the LR
    # follows the 4-epoch curve and stays above zero.
    assert logged[2] == pytest.approx(expected[2], abs=1e-6)
    assert logged[3] == pytest.approx(expected[3], abs=1e-6)
    assert all(lr > 0 for lr in logged)


def test_early_stopping_ends_training_before_the_epoch_limit(project):
    # A zero learning rate can't improve after the first epoch.
    fit(make_config(epochs=10, lr=0.0, early_stopping_patience=2), torch.device("cpu"))
    assert len(read_log()) == 3


def test_load_config_applies_overrides_and_rejects_unknown_keys(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("train:\n  batch_size: 8\n")
    assert load_config(path, ["train.batch_size=4"])["train"]["batch_size"] == 4
    with pytest.raises(KeyError):
        load_config(path, ["train.batch_sizee=4"])
    with pytest.raises(ValueError):
        load_config(path, ["train.batch_size"])


def test_repeated_set_flags_keep_every_override():
    args = build_parser().parse_args(
        ["--set", "data.num_workers=4", "--set", "train.batch_size=4", "train.lr=0.01"]
    )
    assert args.overrides == ["data.num_workers=4", "train.batch_size=4", "train.lr=0.01"]
