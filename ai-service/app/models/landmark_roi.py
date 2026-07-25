"""
MaternIn AI Service — MediaPipe Face Mesh Landmark & ROI Auto-Crop
====================================================================
Menggunakan MediaPipe Face Mesh untuk:
1. Mendeteksi wajah dalam gambar
2. Mengekstraksi landmark mata (palpebral conjunctiva)
3. Auto-crop ROI area konjungtiva untuk input ke model anemia CV

Kasus gagal (wajah tidak terdeteksi, pencahayaan buruk) ditangani
dengan mengembalikan None + pesan error deskriptif.
"""

from __future__ import annotations

import io
import logging
from typing import Any

import numpy as np

logger = logging.getLogger("maternin.ai.landmark_roi")

# MediaPipe Face Mesh landmark indices untuk area mata bawah (palpebral conjunctiva)
# Ref: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
LEFT_EYE_LOWER = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_EYE_LOWER = [362, 382, 381, 380, 374, 373, 390, 249, 263]
LEFT_EYE_UPPER = [246, 161, 160, 159, 158, 157, 173]
RIGHT_EYE_UPPER = [466, 388, 387, 386, 385, 384, 398]

# Padding ratio untuk crop ROI (0.3 = 30% extra di setiap sisi)
ROI_PADDING_RATIO = 0.3
MIN_ROI_SIZE = 32  # Minimum pixel size untuk ROI yang valid


def _try_import_mediapipe():
    """Lazy import MediaPipe — hanya load saat benar-benar dipakai."""
    try:
        import mediapipe as mp
        return mp
    except ImportError:
        logger.warning(
            "MediaPipe not installed — landmark ROI will use fallback mode. "
            "Install with: pip install mediapipe"
        )
        return None


def _try_import_pil():
    """Lazy import PIL — hanya load saat benar-benar dipakai."""
    try:
        from PIL import Image
        return Image
    except ImportError:
        logger.warning("Pillow not installed — install with: pip install Pillow")
        return None


async def download_image(image_url: str) -> bytes | None:
    """Download gambar dari URL menggunakan httpx."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(image_url)
            if response.status_code == 200:
                return response.content
            logger.warning(f"Failed to download image: HTTP {response.status_code}")
            return None
    except Exception as exc:
        logger.warning(f"Failed to download image: {type(exc).__name__}: {exc}")
        return None


def extract_conjunctiva_roi(
    image_bytes: bytes,
    eye: str = "both",
) -> dict[str, Any]:
    """
    Ekstraksi ROI area konjungtiva palpebra dari gambar wajah.

    Args:
        image_bytes: Raw bytes gambar (JPEG/PNG).
        eye: "left", "right", atau "both" (default).

    Returns:
        dict dengan:
          - "success": bool
          - "roi_images": list of numpy arrays (cropped ROI)
          - "error": str (jika gagal)
          - "landmarks_detected": bool
    """
    Image = _try_import_pil()
    mp = _try_import_mediapipe()

    if Image is None or mp is None:
        # Fallback: kembalikan gambar asli tanpa crop
        return _fallback_roi(image_bytes, "Dependencies not available (MediaPipe/Pillow)")

    try:
        # Load gambar
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_array = np.array(image)
        h, w = img_array.shape[:2]

        if h < 50 or w < 50:
            return _fallback_roi(image_bytes, "Image too small (minimum 50x50 pixels)")

        # Jalankan Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
        ) as face_mesh:
            results = face_mesh.process(img_array)

        if not results.multi_face_landmarks:
            return _fallback_roi(image_bytes, "No face detected — pastikan wajah terlihat jelas")

        face_landmarks = results.multi_face_landmarks[0]
        roi_images = []

        # Crop area mata
        eyes_to_process = []
        if eye in ("left", "both"):
            eyes_to_process.append(("left", LEFT_EYE_LOWER + LEFT_EYE_UPPER))
        if eye in ("right", "both"):
            eyes_to_process.append(("right", RIGHT_EYE_LOWER + RIGHT_EYE_UPPER))

        for eye_name, landmarks_indices in eyes_to_process:
            points = []
            for idx in landmarks_indices:
                lm = face_landmarks.landmark[idx]
                px, py = int(lm.x * w), int(lm.y * h)
                points.append((px, py))

            # Bounding box + padding
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            roi_w = x_max - x_min
            roi_h = y_max - y_min
            pad_x = int(roi_w * ROI_PADDING_RATIO)
            pad_y = int(roi_h * ROI_PADDING_RATIO)

            x1 = max(0, x_min - pad_x)
            y1 = max(0, y_min - pad_y)
            x2 = min(w, x_max + pad_x)
            y2 = min(h, y_max + pad_y)

            if (x2 - x1) < MIN_ROI_SIZE or (y2 - y1) < MIN_ROI_SIZE:
                logger.warning(f"ROI too small for {eye_name} eye, skipping")
                continue

            roi = img_array[y1:y2, x1:x2]
            roi_images.append(roi)

        if not roi_images:
            return _fallback_roi(image_bytes, "Eye ROI extraction failed — area terlalu kecil")

        return {
            "success": True,
            "roi_images": roi_images,
            "error": None,
            "landmarks_detected": True,
        }

    except Exception as exc:
        logger.warning(f"Landmark extraction failed: {type(exc).__name__}: {exc}")
        return _fallback_roi(image_bytes, f"Processing error: {exc}")


def _fallback_roi(image_bytes: bytes, reason: str) -> dict[str, Any]:
    """Fallback ketika landmark extraction gagal — gunakan gambar asli."""
    logger.info(f"Using fallback ROI: {reason}")

    Image = _try_import_pil()
    if Image:
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_array = np.array(image)
            # Center crop 50% sebagai fallback ROI
            h, w = img_array.shape[:2]
            ch, cw = h // 4, w // 4
            roi = img_array[ch:h - ch, cw:w - cw]
            return {
                "success": False,
                "roi_images": [roi],
                "error": reason,
                "landmarks_detected": False,
            }
        except Exception:
            pass

    return {
        "success": False,
        "roi_images": [],
        "error": reason,
        "landmarks_detected": False,
    }
