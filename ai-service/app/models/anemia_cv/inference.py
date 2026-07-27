"""
MaternIn AI Service — Anemia CV Detection Inference
=====================================================
Pipeline inferensi: Download gambar → MediaPipe landmark ROI → Model prediksi.
Memuat model ONNX/Keras saat startup, fallback ke mock jika belum tersedia.
"""

from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger("maternin.ai.anemia_cv")

# ── State model ──────────────────────────────────────────────────────────
_model = None
_model_type = None  # "onnx" | "keras" | None
_model_loaded = False
_is_mock = True  # True until a real model is successfully loaded

ONNX_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "model_artifacts", "anemia_convnext_tiny_v2_real.onnx"
)
KERAS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "datasets",
    "anemia_conjunctiva", "HemaVision-Anemia-Triage", "models", "mobilenet_final.keras"
)


def load_model(artifact_path: str | None = None) -> None:
    """
    Load model anemia CV. Prioritas: ONNX > Keras > Mock.
    """
    global _model, _model_type, _model_loaded, _is_mock

    # Priority 1: ONNX
    onnx_path = artifact_path or ONNX_PATH
    if os.path.exists(onnx_path):
        try:
            import onnxruntime as ort
            _model = ort.InferenceSession(onnx_path)
            _model_type = "onnx"
            _model_loaded = True
            _is_mock = False
            logger.info(f"✅ Anemia CV model loaded (ONNX) from {onnx_path}")
            return
        except ImportError:
            logger.warning("onnxruntime not installed, trying Keras fallback")
        except Exception as exc:
            logger.warning(f"Failed to load ONNX model: {exc}")

    # Priority 2: Keras (.keras)
    if os.path.exists(KERAS_PATH):
        try:
            import tensorflow as tf
            _model = tf.keras.models.load_model(KERAS_PATH)
            _model_type = "keras"
            _model_loaded = True
            _is_mock = False
            logger.info(f"✅ Anemia CV model loaded (Keras) from {KERAS_PATH}")
            return
        except ImportError:
            logger.warning("TensorFlow not installed, Keras fallback unavailable")
        except Exception as exc:
            logger.warning(f"Failed to load Keras model: {exc}")

    # Priority 3: Mock
    logger.warning(
        "Anemia CV: using MOCK mode. "
        "Train with: python app/training/anemia_cv_train.py"
    )
    _model_loaded = False


def is_mock_mode() -> bool:
    """Return True if model is running in mock/placeholder mode (no real inference)."""
    return _is_mock


async def predict_anemia(image_url: str | None = None) -> float | None:
    """
    Prediksi probabilitas anemia dari gambar konjungtiva (0.0 - 1.0).

    Pipeline (jika model tersedia):
      1. Download gambar dari image_url
      2. MediaPipe Face Mesh → extract palpebral conjunctiva ROI
      3. Preprocess ROI → Model inference → probabilitas

    Returns:
        float | None: probabilitas anemia, None jika tidak ada gambar.
    """
    if not image_url:
        return None

    if not _model_loaded:
        logger.warning(
            "⚠️ Anemia CV (MOCK): model not loaded, returning placeholder 0.25. "
            "This is NOT a real prediction. "
            "Train and deploy model to get actual inference results. "
            "See: python app/training/anemia_cv_train.py"
        )
        return 0.25  # Mock placeholder — not a real prediction

    try:
        # Step 1: Download gambar
        from app.models.landmark_roi import download_image, extract_conjunctiva_roi

        image_bytes = await download_image(image_url)
        if not image_bytes:
            logger.warning(f"Failed to download image: {image_url[:50]}")
            return None

        # Step 2: Extract ROI
        roi_result = extract_conjunctiva_roi(image_bytes)
        if not roi_result["roi_images"]:
            logger.warning(f"No ROI extracted: {roi_result.get('error')}")
            return None

        roi_image = roi_result["roi_images"][0]

        # Step 3: Preprocess
        preprocessed = _preprocess_roi(roi_image)

        # Step 4: Inference
        if _model_type == "onnx":
            prob = _infer_onnx(preprocessed)
        elif _model_type == "keras":
            prob = _infer_keras(preprocessed)
        else:
            return 0.25

        logger.info(f"Anemia CV inference: prob={prob:.4f}")
        return round(prob, 4)

    except Exception as exc:
        logger.error(f"Anemia CV inference error: {type(exc).__name__}: {exc}")
        return None


def _preprocess_roi(roi: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """Preprocess ROI image untuk model input."""
    from PIL import Image

    img = Image.fromarray(roi).resize(target_size)
    arr = np.array(img, dtype=np.float32) / 255.0

    # Normalize (ImageNet mean/std)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    arr = (arr - mean) / std

    # Add batch dimension: (1, H, W, C) for Keras or (1, C, H, W) for ONNX
    return arr


def _infer_onnx(preprocessed: np.ndarray) -> float:
    """Run ONNX inference."""
    # ONNX expects NCHW format
    if preprocessed.ndim == 3:
        preprocessed = np.expand_dims(preprocessed, axis=0)

    # Try NCHW first
    if preprocessed.shape[-1] == 3:
        preprocessed = np.transpose(preprocessed, (0, 3, 1, 2))

    preprocessed = preprocessed.astype(np.float32)

    input_name = _model.get_inputs()[0].name
    output = _model.run(None, {input_name: preprocessed})

    # Sigmoid / softmax output
    logits = output[0][0]
    if len(logits) == 2:
        # Binary classification [normal, anemia]
        prob = float(logits[1])
    elif len(logits) == 1:
        # Single sigmoid output
        prob = float(1 / (1 + np.exp(-logits[0])))
    else:
        prob = float(logits[1]) if len(logits) > 1 else float(logits[0])

    return min(max(prob, 0.0), 1.0)


def _infer_keras(preprocessed: np.ndarray) -> float:
    """Run Keras/TF inference."""
    if preprocessed.ndim == 3:
        preprocessed = np.expand_dims(preprocessed, axis=0)

    preprocessed = preprocessed.astype(np.float32)
    prediction = _model.predict(preprocessed, verbose=0)

    if prediction.shape[-1] == 2:
        prob = float(prediction[0][1])
    else:
        prob = float(prediction[0][0])

    return min(max(prob, 0.0), 1.0)
