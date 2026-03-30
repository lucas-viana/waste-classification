import os
import json
import shutil
import tempfile
from functools import lru_cache
from io import BytesIO
from pathlib import Path
from typing import Annotated, Any

import h5py
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from tensorflow import keras

ROOT_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = Path(
    os.getenv("MODEL_PATH", ROOT_DIR / "backend" / "models" / "modelo_residuos.h5")
)
DEFAULT_CLASS_NAMES = "organic,recyclable"
RESCALE_INPUT = os.getenv("RESCALE_INPUT", "true").lower() == "true"

app = FastAPI(
    title="Waste Classification API",
    version="1.0.0",
    description="Classifies waste images using a TensorFlow/Keras model.",
)

cors_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in cors_origins if origin.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _class_names_from_env() -> list[str]:
    names_raw = os.getenv("WASTE_CLASS_NAMES", DEFAULT_CLASS_NAMES)
    names = [item.strip() for item in names_raw.split(",") if item.strip()]
    return names if names else ["class_0", "class_1"]


def _remove_key_deep(payload: Any, key_to_remove: str) -> None:
    if isinstance(payload, dict):
        payload.pop(key_to_remove, None)
        for value in payload.values():
            _remove_key_deep(value, key_to_remove)
    elif isinstance(payload, list):
        for item in payload:
            _remove_key_deep(item, key_to_remove)


def _load_h5_model_with_legacy_config(model_path: Path) -> Any:
    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as temp_file:
        temp_h5_path = Path(temp_file.name)

    try:
        shutil.copy2(model_path, temp_h5_path)

        with h5py.File(temp_h5_path, "r+") as h5_file:
            model_config = h5_file.attrs.get("model_config")

            if model_config is None:
                raise ValueError("Could not find model_config in the .h5 file.")

            if not isinstance(model_config, str):
                model_config = model_config.decode("utf-8")

            config_dict = json.loads(model_config)
            _remove_key_deep(config_dict, "quantization_config")
            h5_file.attrs["model_config"] = json.dumps(config_dict).encode("utf-8")

        return keras.models.load_model(temp_h5_path, compile=False)
    finally:
        temp_h5_path.unlink(missing_ok=True)


@lru_cache(maxsize=1)
def load_model() -> Any:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file was not found at {MODEL_PATH}. Set MODEL_PATH env var if needed."
        )

    try:
        return keras.models.load_model(MODEL_PATH, compile=False)
    except (TypeError, ValueError):
        if MODEL_PATH.suffix.lower() not in {".h5", ".hdf5"}:
            raise

        # Fallback for legacy H5 files with unsupported config fields in newer Keras.
        return _load_h5_model_with_legacy_config(MODEL_PATH)


def _infer_input_shape(model: Any) -> tuple[int, int, int]:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]

    if not isinstance(input_shape, tuple) or len(input_shape) < 4:
        return 224, 224, 3

    height = input_shape[1] if input_shape[1] is not None else 224
    width = input_shape[2] if input_shape[2] is not None else 224
    channels = input_shape[3] if input_shape[3] is not None else 3
    return int(height), int(width), int(channels)


def _prepare_image(file_bytes: bytes, model: Any) -> np.ndarray:
    target_h, target_w, channels = _infer_input_shape(model)

    try:
        image = Image.open(BytesIO(file_bytes))
    except UnidentifiedImageError as exc:
        raise ValueError("Invalid image file.") from exc

    if channels == 1:
        image = image.convert("L")
    else:
        image = image.convert("RGB")

    image = image.resize((target_w, target_h))
    array = np.asarray(image, dtype=np.float32)

    if channels == 1 and array.ndim == 2:
        array = np.expand_dims(array, axis=-1)

    if RESCALE_INPUT:
        array = array / 255.0

    return np.expand_dims(array, axis=0)


def _to_probabilities(raw_prediction: np.ndarray) -> np.ndarray:
    squeezed = np.squeeze(raw_prediction)

    if np.isscalar(squeezed):
        positive = float(np.clip(squeezed, 0.0, 1.0))
        return np.array([1.0 - positive, positive], dtype=np.float32)

    vector = np.asarray(squeezed, dtype=np.float32).flatten()
    if vector.size == 1:
        positive = float(np.clip(vector[0], 0.0, 1.0))
        return np.array([1.0 - positive, positive], dtype=np.float32)

    # If outputs are logits, convert to softmax probabilities.
    if np.any(vector < 0.0) or not np.isclose(np.sum(vector), 1.0, atol=1e-3):
        exp_values = np.exp(vector - np.max(vector))
        return exp_values / np.sum(exp_values)

    return vector


def _align_class_names(class_names: list[str], total: int) -> list[str]:
    if len(class_names) >= total:
        return class_names[:total]

    generated = [f"class_{index}" for index in range(len(class_names), total)]
    return class_names + generated


@app.get("/api/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/api/model-info")
def model_info() -> dict[str, Any]:
    model = load_model()
    target_h, target_w, channels = _infer_input_shape(model)
    class_names = _class_names_from_env()

    return {
        "model_path": str(MODEL_PATH),
        "input": {"height": target_h, "width": target_w, "channels": channels},
        "class_names": class_names,
        "rescale_input": RESCALE_INPUT,
    }


@app.post(
    "/api/predict",
    responses={400: {"description": "Invalid request payload or image."}},
)
async def predict(file: Annotated[UploadFile, File(...)]) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Missing file name.")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are accepted.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    model = load_model()
    try:
        batch = _prepare_image(image_bytes, model)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    raw_prediction = model.predict(batch, verbose=0)
    probabilities = _to_probabilities(raw_prediction)

    class_names = _align_class_names(_class_names_from_env(), int(probabilities.shape[0]))
    top_index = int(np.argmax(probabilities))
    top_confidence = float(probabilities[top_index])

    return {
        "predicted_class": class_names[top_index],
        "predicted_index": top_index,
        "confidence": round(top_confidence, 6),
        "probabilities": [
            {
                "class_name": class_names[index],
                "probability": round(float(prob), 6),
            }
            for index, prob in enumerate(probabilities)
        ],
    }
