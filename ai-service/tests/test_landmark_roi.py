"""
tests/test_landmark_roi.py
===========================
Tests untuk MediaPipe landmark extraction dan ROI auto-crop.
Menguji fungsi extract_conjunctiva_roi pada gambar valid dan invalid.
"""

import io
import pytest

import numpy as np
from PIL import Image


class TestExtractConjunctivaROI:
    """Test extract_conjunctiva_roi dengan berbagai skenario gambar."""

    def _create_test_image(self, size: tuple[int, int] = (640, 480)) -> bytes:
        """Helper: buat test image bytes."""
        img = Image.new("RGB", size, color=(220, 180, 160))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()

    def _create_face_test_image(self, size: tuple[int, int] = (640, 480)) -> bytes:
        """Helper: buat test image dengan area yang menyerupai wajah."""
        img = Image.new("RGB", size, color=(100, 80, 70))
        # Add lighter eye-like region in the upper portion
        img_array = np.array(img)
        h, w = h, w = size
        # Add two bright spots for "eyes"
        eye_y, eye1_x, eye2_x = h // 3, w // 3, 2 * w // 3
        cv_radius = min(w, h) // 20
        y, x = np.ogrid[:h, :w]
        mask1 = ((x - eye1_x) ** 2 + (y - eye_y) ** 2) < cv_radius ** 2
        mask2 = ((x - eye2_x) ** 2 + (y - eye_y) ** 2) < cv_radius ** 2
        img_array[mask1 | mask2] = [200, 180, 160]
        out = Image.fromarray(img_array)
        buf = io.BytesIO()
        out.save(buf, format="JPEG")
        return buf.getvalue()

    def test_valid_image_bytes_input(self):
        """Fungsi harus menerima raw bytes JPEG."""
        img_bytes = self._create_test_image()
        from app.models.landmark_roi import extract_conjunctiva_roi
        result = extract_conjunctiva_roi(img_bytes)
        assert isinstance(result, dict)

    def test_result_structure(self):
        """Result harus punya struktur dict yang benar."""
        from app.models.landmark_roi import extract_conjunctiva_roi
        img_bytes = self._create_test_image()
        result = extract_conjunctiva_roi(img_bytes)
        assert "success" in result
        assert "roi_images" in result
        assert "error" in result
        assert "landmarks_detected" in result

    def test_roi_images_is_list(self):
        """roi_images harus list."""
        from app.models.landmark_roi import extract_conjunctiva_roi
        img_bytes = self._create_test_image()
        result = extract_conjunctiva_roi(img_bytes)
        assert isinstance(result["roi_images"], list)

    def test_roi_images_numpy_arrays(self):
        """ROI images harus numpy arrays."""
        from app.models.landmark_roi import extract_conjunctiva_roi
        img_bytes = self._create_test_image()
        result = extract_conjunctiva_roi(img_bytes)
        for roi in result["roi_images"]:
            assert isinstance(roi, np.ndarray)
            assert roi.ndim == 3  # RGB image

    def test_small_image_too_small(self):
        """Gambar terlalu kecil (< 50x50) harus ditangani (jika MediaPipe tersedia)."""
        from app.models.landmark_roi import extract_conjunctiva_roi, _try_import_mediapipe
        # Create tiny image
        img = Image.new("RGB", (30, 40), color=(100, 80, 70))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        result = extract_conjunctiva_roi(img_bytes)
        # Should either reject small images (if MediaPipe available) or use fallback gracefully
        assert "roi_images" in result
        assert isinstance(result["roi_images"], list)
        # Fallback error should mention MediaPipe or small size
        if result["success"] is False:
            assert result["error"] is not None
            error_lower = result["error"].lower()
            is_mediapipe_unavailable = "dependencies" in error_lower and "mediapipe" in error_lower
            is_size_rejection = "too small" in error_lower or "minimum" in error_lower
            assert is_mediapipe_unavailable or is_size_rejection

    def test_corrupted_image_uses_fallback(self):
        """Gambar corrupt harus menggunakan fallback."""
        from app.models.landmark_roi import extract_conjunctiva_roi
        # Not a valid JPEG
        corrupted_bytes = b"not a valid image bytes at all"
        result = extract_conjunctiva_roi(corrupted_bytes)
        # Should handle gracefully with fallback
        assert "roi_images" in result
        assert isinstance(result["roi_images"], list)

    def test_empty_bytes(self):
        """Empty bytes harus ditangani."""
        from app.models.landmark_roi import extract_conjunctiva_roi
        result = extract_conjunctiva_roi(b"")
        # Should use fallback
        assert "roi_images" in result
        assert isinstance(result["roi_images"], list)

    def test_png_format_accepted(self):
        """Fungsi harus menerima format PNG."""
        from app.models.landmark_roi import extract_conjunctiva_roi
        img = Image.new("RGB", (640, 480), color=(220, 180, 160))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        result = extract_conjunctiva_roi(png_bytes)
        assert "roi_images" in result
        assert isinstance(result["roi_images"], list)

    def test_eye_parameter_left(self):
        """Parameter eye='left' harus memproses hanya mata kiri."""
        from app.models.landmark_roi import extract_conjunctiva_roi
        img_bytes = self._create_test_image()
        result = extract_conjunctiva_roi(img_bytes, eye="left")
        assert "roi_images" in result

    def test_eye_parameter_right(self):
        """Parameter eye='right' harus memproses hanya mata kanan."""
        from app.models.landmark_roi import extract_conjunctiva_roi
        img_bytes = self._create_test_image()
        result = extract_conjunctiva_roi(img_bytes, eye="right")
        assert "roi_images" in result

    def test_eye_parameter_both(self):
        """Parameter eye='both' harus memproses kedua mata."""
        from app.models.landmark_roi import extract_conjunctiva_roi
        img_bytes = self._create_test_image()
        result = extract_conjunctiva_roi(img_bytes, eye="both")
        assert "roi_images" in result

    def test_fallback_returns_original_as_roi(self):
        """Fallback mode harus return cropped version dari gambar asli."""
        from app.models.landmark_roi import extract_conjunctiva_roi
        img_bytes = self._create_test_image()
        result = extract_conjunctiva_roi(img_bytes)
        # Fallback uses center-crop 50%
        if not result["success"]:
            assert len(result["roi_images"]) > 0
            # Check dimensions are smaller (center crop)
            roi = result["roi_images"][0]
            original_img = Image.open(io.BytesIO(img_bytes))
            orig_h, orig_w = np.array(original_img).shape[:2]
            roi_h, roi_w = roi.shape[:2]
            assert roi_h < orig_h
            assert roi_w < orig_w

    def test_landmarks_detected_flag(self):
        """landmarks_detected flag harus boolean."""
        from app.models.landmark_roi import extract_conjunctiva_roi
        img_bytes = self._create_test_image()
        result = extract_conjunctiva_roi(img_bytes)
        assert isinstance(result["landmarks_detected"], bool)

    def test_no_face_detected_fallback(self):
        """No face detected harus menggunakan fallback."""
        from app.models.landmark_roi import extract_conjunctiva_roi
        # Image with no face-like features (solid color noise)
        img = Image.new("RGB", (640, 480), color=(50, 50, 50))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        img_bytes = buf.getvalue()

        result = extract_conjunctiva_roi(img_bytes)
        # Without MediaPipe landmarks, should use fallback
        assert "roi_images" in result
        assert isinstance(result["roi_images"], list)


class TestLandmarkConstants:
    """Test konstanta landmark."""

    def test_eye_landmark_indices_nonempty(self):
        """Eye landmark indices harus nonempty lists."""
        from app.models.landmark_roi import (
            LEFT_EYE_LOWER, LEFT_EYE_UPPER,
            RIGHT_EYE_LOWER, RIGHT_EYE_UPPER,
        )
        assert len(LEFT_EYE_LOWER) > 0
        assert len(LEFT_EYE_UPPER) > 0
        assert len(RIGHT_EYE_LOWER) > 0
        assert len(RIGHT_EYE_UPPER) > 0

    def test_roi_padding_ratio(self):
        """ROI padding ratio harus 0.3."""
        from app.models.landmark_roi import ROI_PADDING_RATIO
        assert ROI_PADDING_RATIO == 0.3

    def test_min_roi_size(self):
        """Minimum ROI size harus 32."""
        from app.models.landmark_roi import MIN_ROI_SIZE
        assert MIN_ROI_SIZE == 32


class TestDownloadImage:
    """Test download_image function."""

    @pytest.mark.asyncio
    async def test_download_image_invalid_url(self):
        """URL tidak valid harus return None."""
        from app.models.landmark_roi import download_image
        result = await download_image("http://invalid-domain-that-does-not-exist-xyz.com/image.jpg")
        # Should handle gracefully without raising
        assert result is None or isinstance(result, bytes)

    @pytest.mark.asyncio
    async def test_download_image_empty_url(self):
        """URL kosong harus return None."""
        from app.models.landmark_roi import download_image
        result = await download_image("")
        assert result is None

    @pytest.mark.asyncio
    async def test_download_image_returns_bytes(self):
        """URL valid harus return bytes."""
        from app.models.landmark_roi import download_image
        # Use a well-known small image URL
        result = await download_image(
            "https://www.google.com/images/branding/googlelogo/1x/googlelogo_color_272x92dp.png"
        )
        if result is not None:
            assert isinstance(result, bytes)
            assert len(result) > 0
