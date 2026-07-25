"""
Pydantic Schemas — Trend Predict Endpoint
===========================================
Kontrak request/response untuk POST /api/v1/trend/predict (P1).
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class TrendDirection(str, Enum):
    naik = "naik"
    stabil = "stabil"
    turun = "turun"


class ScoreHistoryEntry(BaseModel):
    """Sub-model: satu titik data histori skor."""
    aggregate_score: float = Field(..., ge=0, le=100, description="Skor agregat risiko")
    created_at: datetime = Field(..., description="Timestamp penilaian (UTC)")


class TrendPredictRequest(BaseModel):
    """Request body untuk POST /api/v1/trend/predict."""

    pregnancy_profile_id: str = Field(
        ..., description="UUID profil kehamilan pasien"
    )
    score_history: list[ScoreHistoryEntry] = Field(
        ...,
        min_length=2,
        description="Histori skor agregat (minimal 2 titik data)",
    )


class TrendPredictResponse(BaseModel):
    """Response body dari POST /api/v1/trend/predict."""

    trend_direction: TrendDirection = Field(
        ..., description="Arah tren: naik / stabil / turun"
    )
    predicted_badge_in_days: int | None = Field(
        default=None,
        description="Estimasi hari hingga badge berubah (null jika stabil)",
    )
    predicted_badge: str | None = Field(
        default=None,
        description="Badge risiko yang diprediksi (hijau/kuning/merah)",
    )
    confidence_note: str = Field(
        default="",
        description="Catatan tingkat kepercayaan prediksi",
    )
