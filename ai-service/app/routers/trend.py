"""
MaternIn AI Service — Trend Predict Router
============================================
POST /api/v1/trend/predict (P1)

Prediksi tren risiko kehamilan dari histori aggregate_score.
Menggunakan regresi linear sederhana — PRD eksplisit minta jangan over-engineer.
"""

from __future__ import annotations

import logging
import math

from fastapi import APIRouter, Depends

from app.core.auth import get_request_id, verify_internal_token
from app.schemas.trend import (
    ScoreHistoryEntry,
    TrendDirection,
    TrendPredictRequest,
    TrendPredictResponse,
)

logger = logging.getLogger("maternin.ai.router.trend")

router = APIRouter(
    prefix="/api/v1/trend",
    tags=["Trend Prediction"],
    dependencies=[Depends(verify_internal_token)],
)

# Badge thresholds (konsisten dengan XGBoost aggregator)
BADGE_THRESHOLDS = {
    "hijau": (0, 30),
    "kuning": (30, 70),
    "merah": (70, 100),
}


def _score_to_badge(score: float) -> str:
    """Konversi skor ke badge risiko."""
    if score >= 70:
        return "merah"
    elif score >= 30:
        return "kuning"
    return "hijau"


def _linear_regression(
    entries: list[ScoreHistoryEntry],
) -> dict:
    """
    Simple linear regression: y = a + b*x
    x = hari relatif dari titik pertama
    y = aggregate_score
    """
    sorted_entries = sorted(entries, key=lambda e: e.created_at)

    # Convert timestamps to days relative to first entry
    t0 = sorted_entries[0].created_at
    x = [(e.created_at - t0).total_seconds() / 86400 for e in sorted_entries]
    y = [e.aggregate_score for e in sorted_entries]

    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    sum_x2 = sum(xi ** 2 for xi in x)

    # Avoid division by zero
    denom = n * sum_x2 - sum_x ** 2
    if abs(denom) < 1e-10:
        return {
            "slope": 0.0,
            "intercept": sum_y / n if n > 0 else 0.0,
            "current_score": y[-1] if y else 0.0,
            "days_span": x[-1] if x else 0.0,
        }

    b = (n * sum_xy - sum_x * sum_y) / denom  # slope
    a = (sum_y - b * sum_x) / n  # intercept

    return {
        "slope": b,
        "intercept": a,
        "current_score": y[-1],
        "days_span": x[-1],
    }


def _predict_trend(
    entries: list[ScoreHistoryEntry],
) -> TrendPredictResponse:
    """
    Hitung arah tren dan prediksi perubahan badge.
    """
    reg = _linear_regression(entries)
    slope = reg["slope"]
    current_score = reg["current_score"]
    current_badge = _score_to_badge(current_score)
    n_points = len(entries)

    # Determine trend direction
    # Slope > 1 point/day → naik, < -1 → turun, else stabil
    if slope > 1.0:
        direction = TrendDirection.naik
    elif slope < -1.0:
        direction = TrendDirection.turun
    else:
        direction = TrendDirection.stabil

    # Predict badge change
    predicted_badge = None
    predicted_days = None

    if direction == TrendDirection.naik:
        # Predict when score crosses next threshold
        if current_badge == "hijau" and slope > 0:
            days_to_kuning = (30 - current_score) / slope
            if 0 < days_to_kuning <= 30:
                predicted_badge = "kuning"
                predicted_days = max(1, math.ceil(days_to_kuning))
        elif current_badge == "kuning" and slope > 0:
            days_to_merah = (70 - current_score) / slope
            if 0 < days_to_merah <= 30:
                predicted_badge = "merah"
                predicted_days = max(1, math.ceil(days_to_merah))
        elif current_badge == "merah":
            predicted_badge = "merah"
            predicted_days = 0

    elif direction == TrendDirection.turun:
        if current_badge == "merah" and slope < 0:
            days_to_kuning = (current_score - 70) / abs(slope)
            if 0 < days_to_kuning <= 30:
                predicted_badge = "kuning"
                predicted_days = max(1, math.ceil(days_to_kuning))
        elif current_badge == "kuning" and slope < 0:
            days_to_hijau = (current_score - 30) / abs(slope)
            if 0 < days_to_hijau <= 30:
                predicted_badge = "hijau"
                predicted_days = max(1, math.ceil(days_to_hijau))

    # Confidence note
    if n_points <= 2:
        confidence = (
            f"Berdasarkan {n_points} titik data — prediksi ini sangat awal "
            "dan perlu data lebih banyak untuk validasi."
        )
    elif n_points <= 5:
        confidence = (
            f"Berdasarkan {n_points} titik data, interpretasi tetap perlu validasi bidan."
        )
    else:
        confidence = (
            f"Berdasarkan {n_points} titik data dengan tren {direction.value}. "
            "Tetap konsultasikan interpretasi dengan bidan."
        )

    return TrendPredictResponse(
        trend_direction=direction,
        predicted_badge_in_days=predicted_days,
        predicted_badge=predicted_badge,
        confidence_note=confidence,
    )


@router.post(
    "/predict",
    response_model=TrendPredictResponse,
    summary="Prediksi tren risiko kehamilan",
    description=(
        "Menganalisis histori aggregate_score untuk memprediksi arah tren risiko "
        "dan kemungkinan perubahan badge dalam beberapa hari ke depan."
    ),
)
async def trend_predict(
    request: TrendPredictRequest,
    request_id: str | None = Depends(get_request_id),
) -> TrendPredictResponse:
    """Prediksi tren dari histori skor."""
    logger.info(
        f"[{request_id}] Trend predict: "
        f"profile={request.pregnancy_profile_id}, "
        f"data_points={len(request.score_history)}"
    )

    result = _predict_trend(request.score_history)

    logger.info(
        f"[{request_id}] Trend result: "
        f"direction={result.trend_direction.value}, "
        f"predicted_badge={result.predicted_badge}, "
        f"days={result.predicted_badge_in_days}"
    )

    return result
