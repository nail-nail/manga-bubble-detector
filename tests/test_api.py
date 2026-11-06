import io

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image

from src.api import app
from src.model import UNet


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    """A real checkpoint with untrained weights. These tests cover the API, not model quality."""
    path = tmp_path_factory.mktemp("ckpt") / "model.pt"
    torch.manual_seed(0)
    model = UNet(base_channels=4)
    config = {
        "data": {"image_size": 64},
        "model": {"in_channels": 1, "base_channels": 4},
        "postprocess": {"threshold": 0.5, "open_kernel": 3, "min_area": 1, "erode_px": 0},
    }
    torch.save({"model": model.state_dict(), "config": config}, path)
    return path


@pytest.fixture
def client(checkpoint, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(checkpoint))
    with TestClient(app) as test_client:
        yield test_client


def encode(fmt: str = "PNG", size=(120, 200), mode="L") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, color=255 if mode == "L" else (255, 255, 255)).save(buffer, format=fmt)
    return buffer.getvalue()


def upload(client, path, data, name="page.png", content_type="image/png"):
    return client.post(path, files={"file": (name, data, content_type)})


def test_health_reports_status_and_model_version(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert len(body["model_version"]) == 12


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_predict_returns_boxes_and_timing(client, fmt):
    response = upload(client, "/predict", encode(fmt))
    assert response.status_code == 200
    body = response.json()
    assert isinstance(body["boxes"], list)
    assert body["inference_ms"] > 0


def test_predicted_boxes_use_original_page_coordinates(client, monkeypatch):
    # Force a known probability map so coordinates can be checked exactly.
    predictor = app.state.predictor
    original = predictor.model

    class Fixed(torch.nn.Module):
        def forward(self, x):
            logits = torch.full_like(x, -10.0)
            logits[..., 20:44, 20:44] = 10.0
            return logits

    monkeypatch.setattr(predictor, "model", Fixed())
    # A 320x640 page in a 64 canvas: scale 0.1, with 16 px of padding on each side.
    response = upload(client, "/predict", encode(size=(320, 640)))
    predictor.model = original
    boxes = response.json()["boxes"]
    assert len(boxes) == 1
    box = boxes[0]
    assert (box["x"], box["y"], box["w"], box["h"]) == (40, 200, 240, 240)
    assert 0.99 < box["score"] <= 1.0


def test_predict_image_returns_a_png_of_the_same_size(client):
    response = upload(client, "/predict/image", encode(size=(120, 200)))
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    decoded = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape[:2] == (200, 120)


def test_model_is_loaded_once_and_not_per_request(client):
    first = app.state.predictor
    upload(client, "/predict", encode())
    upload(client, "/predict", encode())
    assert app.state.predictor is first


def test_rejects_a_non_image_even_with_an_image_name_and_type(client):
    response = upload(client, "/predict", b"this is not an image")
    assert response.status_code == 415


def test_rejects_formats_outside_the_allow_list(client):
    response = upload(client, "/predict", encode("GIF"), name="page.gif", content_type="image/gif")
    assert response.status_code == 415


def test_accepts_a_real_image_regardless_of_its_declared_type(client):
    response = upload(client, "/predict", encode(), name="page.txt", content_type="text/plain")
    assert response.status_code == 200


def test_rejects_an_upload_over_the_size_limit(client, monkeypatch):
    monkeypatch.setenv("MAX_UPLOAD_MB", "1")
    response = upload(client, "/predict", b"x" * (1024 * 1024 + 1))
    assert response.status_code == 413


def test_rejects_an_image_with_too_many_pixels(client, monkeypatch):
    monkeypatch.setattr("src.api.MAX_PIXELS", 100)
    response = upload(client, "/predict", encode(size=(20, 20)))
    assert response.status_code == 413


def test_rejects_a_truncated_image_without_crashing(client):
    data = encode("PNG", size=(300, 300))
    response = upload(client, "/predict", data[:40])
    assert response.status_code in (400, 415)


def test_missing_file_field_returns_422(client):
    assert client.post("/predict").status_code == 422


def test_startup_fails_clearly_when_the_checkpoint_is_missing(monkeypatch, tmp_path):
    monkeypatch.setenv("MODEL_PATH", str(tmp_path / "nope.pt"))
    with pytest.raises(RuntimeError, match="checkpoint not found"):
        with TestClient(app):
            pass
