"""FastAPI service: upload a manga page, get back bubble boxes or an annotated image.

Run locally:  uvicorn src.api:app --port 8000   then open /docs for Swagger UI.
Settings:     MODEL_PATH (default checkpoints/best.pt), MAX_UPLOAD_MB (default 10).
"""

from __future__ import annotations

import io
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from PIL import Image, UnidentifiedImageError

from src.inference import Predictor
from src.postprocess import draw_boxes

ALLOWED_FORMATS = {"PNG", "JPEG"}
MAX_PIXELS = 50_000_000  # Rejects "decompression bomb" files that are small but huge once decoded.
READ_CHUNK = 1024 * 1024


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Loading once here means requests never pay the cost of reading the model.
    path = os.environ.get("MODEL_PATH", "checkpoints/best.pt")
    if not os.path.exists(path):
        raise RuntimeError(f"Model checkpoint not found at {path}. Set MODEL_PATH or mount the file.")
    app.state.predictor = Predictor(path, device="cpu")
    yield


app = FastAPI(title="Manga Speech-Bubble Detector", version="0.1.0", lifespan=lifespan)


def decode_upload(file: UploadFile) -> np.ndarray:
    """Read and validate an upload, returning the page as a BGR array."""
    max_bytes = int(os.environ.get("MAX_UPLOAD_MB", "10")) * 1024 * 1024
    # Read a chunk at a time, so an oversized upload is rejected without being buffered whole.
    buffer = bytearray()
    while chunk := file.file.read(READ_CHUNK):
        buffer.extend(chunk)
        if len(buffer) > max_bytes:
            raise HTTPException(413, f"File too large. The limit is {max_bytes // (1024 * 1024)} MB.")
    data = bytes(buffer)

    # Trust the file's actual bytes rather than its name or declared content type.
    try:
        with Image.open(io.BytesIO(data)) as probe:  # Reads the header only.
            image_format, (width, height) = probe.format, probe.size
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(415, "Unsupported file. Upload a PNG or JPEG image.") from None
    if image_format not in ALLOWED_FORMATS:
        raise HTTPException(415, "Unsupported file. Upload a PNG or JPEG image.")
    if width * height > MAX_PIXELS:
        raise HTTPException(413, f"Image too large. The limit is {MAX_PIXELS:,} pixels.")

    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(400, "The image file is corrupted and could not be decoded.")
    return image


@app.get("/health")
def health(request: Request) -> dict:
    return {"status": "ok", "model_version": request.app.state.predictor.version}


# Plain `def` endpoints: FastAPI runs them in a thread pool, so slow CPU inference
# doesn't block the event loop.
@app.post("/predict")
def predict(request: Request, file: UploadFile = File(...)) -> dict:
    image = decode_upload(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes, elapsed_ms = request.app.state.predictor.predict(gray)
    return {"boxes": [box.to_dict() for box in boxes], "inference_ms": round(elapsed_ms, 1)}


@app.post(
    "/predict/image",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}, "description": "Page with boxes drawn on it"}},
)
def predict_image(request: Request, file: UploadFile = File(...)) -> Response:
    image = decode_upload(file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes, _ = request.app.state.predictor.predict(gray)
    ok, encoded = cv2.imencode(".png", draw_boxes(image, boxes))
    if not ok:
        raise HTTPException(500, "Could not encode the output image.")
    return Response(content=encoded.tobytes(), media_type="image/png")
