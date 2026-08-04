"""
MaternIn AI Service — Preeclampsia Logistic Regression Inference
==================================================================
Wrapper inferensi untuk model LR preeklampsia.
Memuat model .pkl saat startup, fallback ke heuristik jika file belum ada.

PENTING: fitur model sekarang 29 (termasuk engineered features dari training).
Lihat `_engineer_features()` — harus match training script `risk_aggregator_improved.py`.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

logger = logging.getLogger("maternin.ai.preeclampsia_lr")

# ── State model ──────────────────────────────────────────────────────────
_pipeline = None
_model_loaded = False

ARTIFACT_FILENAME = "preeclampsia_lr_v1.pkl"


def _artifact_path() -> str:
    """
    Return path ke .pkl preeclampsia LR model.
    Resolved dari HF Hub cache (production) atau folder lokal (dev).
    """
    from app.core.artifact_loader import ensure_model_artifacts
    return str(ensure_model_artifacts() / ARTIFACT_FILENAME)

# Mapping protein urine ke encoded value (harus konsisten dengan training)
PROTEIN_URINE_MAP = {
    "negatif": 0, "trace": 1, "positif_1": 2, "positif_2": 3,
    "positif_3": 4, "positif_4": 5,
    "positif_ringan": 1, "positif": 2, "positif_kuat": 4,
    "+1": 2, "+2": 3, "+3": 4, "+4": 5,
}

# Order of features after engineering (must match training script)
FEATURE_COLUMNS = [
    "systolic_bp", "diastolic_bp", "age", "gestational_age_weeks", "bmi",
    "has_hypertension_history", "protein_urine_encoded", "has_preeclampsia_history",
    "systolic_diastolic_product", "pulse_pressure", "mean_arterial_pressure", "bp_severity",
    "age_bp_interaction", "is_extreme_age", "is_high_age", "is_young",
    "heart_rate_abnormal", "heart_rate_severity", "fever", "hypothermia", "bs_abnormal", "bs_severity",
    "severe_hypertension", "moderate_hypertension", "abnormal_vitals_count",
    "age_bp_severity", "multi_system_abnormality",
    "systolic_squared", "diastolic_squared",
]


def _engineer_features(
    systolic: float,
    diastolic: float,
    age: float,
    gestational_age_weeks: float,
    bmi: float,
    has_hypertension_history: bool,
    protein_urine_encoded: int,
    has_preeclampsia_history: bool,
    # vitals (use defaults if not provided)
    heart_rate: float = 80,
    body_temp: float = 37.0,
    bs: float = 10.0,
) -> np.ndarray:
    """Build 29-feature vector matching training script."""
    features = {
        "systolic_bp": systolic,
        "diastolic_bp": diastolic,
        "age": age,
        "gestational_age_weeks": gestational_age_weeks,
        "bmi": bmi,
        "has_hypertension_history": int(has_hypertension_history),
        "protein_urine_encoded": protein_urine_encoded,
        "has_preeclampsia_history": int(has_preeclampsia_history),
        # engineered
        "systolic_diastolic_product": systolic * diastolic,
        "pulse_pressure": systolic - diastolic,
        "mean_arterial_pressure": diastolic + (systolic - diastolic) / 3,
        "bp_severity": (systolic - 120) / 30 + (diastolic - 80) / 20,
        "age_bp_interaction": age * systolic / 100,
        "is_extreme_age": int((age < 18) or (age > 35)),
        "is_high_age": int(age >= 35),
        "is_young": int(age < 20),
        "heart_rate_abnormal": int((heart_rate < 60) or (heart_rate > 100)),
        "heart_rate_severity": abs(heart_rate - 80) / 20,
        "fever": int(body_temp > 38.0),
        "hypothermia": int(body_temp < 36.0),
        "bs_abnormal": int((bs < 7) or (bs > 14)),
        "bs_severity": abs(bs - 10) / 5,
        "severe_hypertension": int((systolic >= 160) or (diastolic >= 110)),
        "moderate_hypertension": int(
            ((systolic >= 140) and (systolic < 160)) or
            ((diastolic >= 90) and (diastolic < 110))
        ),
        "abnormal_vitals_count": (
            int((heart_rate < 60) or (heart_rate > 100))
            + int(body_temp > 38.0)
            + int(body_temp < 36.0)
            + int((bs < 7) or (bs > 14))
        ),
        "age_bp_severity": ((systolic - 120) / 30 + (diastolic - 80) / 20) * int(age >= 35),
        "multi_system_abnormality": int(
            ((systolic >= 140) and (systolic < 160)) or
            ((diastolic >= 90) and (diastolic < 110))
        ) * (
            int((heart_rate < 60) or (heart_rate > 100))
            + int(body_temp > 38.0)
            + int(body_temp < 36.0)
            + int((bs < 7) or (bs > 14))
        ),
        "systolic_squared": systolic ** 2 / 1000,
        "diastolic_squared": diastolic ** 2 / 1000,
    }
    return np.array([[features[col] for col in FEATURE_COLUMNS]])


def load_model(artifact_path: str | None = None) -> None:
    """
    Load model Logistic Regression dari file .pkl.
    Dipanggil saat FastAPI startup (lifespan event).
    """
    global _pipeline, _model_loaded

    path = artifact_path or _artifact_path()
    if not os.path.exists(path):
        logger.warning(
            f"Preeclampsia LR model not found at {path}. "
            f"Using fallback heuristic mode. "
            f"Train with: python app/training/risk_aggregator_improved.py"
        )
        _model_loaded = False
        return

    try:
        import joblib
        _pipeline = joblib.load(path)
        _model_loaded = True
        logger.info(f"✅ Preeclampsia LR model loaded from {path}")
    except Exception as exc:
        logger.error(f"Failed to load preeclampsia model: {exc}")
        _model_loaded = False


def predict_preeclampsia(
    systolic: int | None = None,
    diastolic: int | None = None,
    protein_urine: str | None = None,
    has_preeclampsia_history: bool = False,
    has_hypertension_history: bool = False,
    age: int | float = 25,
    gestational_age_weeks: int | float = 28,
    bmi: float = 24.0,
    heart_rate: float | None = None,
    body_temp: float | None = None,
    bs: float | None = None,
    **kwargs: Any,
) -> float:
    """
    Prediksi probabilitas preeklampsia (0.0 - 1.0).
    """
    if _model_loaded and _pipeline is not None:
        return _predict_with_model(
            systolic, diastolic, protein_urine,
            has_preeclampsia_history, has_hypertension_history,
            age, gestational_age_weeks, bmi,
            heart_rate, body_temp, bs,
        )
    else:
        return _predict_heuristic(
            systolic, diastolic, protein_urine,
            has_preeclampsia_history,
        )


def _predict_with_model(
    systolic, diastolic, protein_urine,
    has_preeclampsia_history, has_hypertension_history,
    age, gestational_age_weeks, bmi,
    heart_rate, body_temp, bs,
) -> float:
    """Inferensi menggunakan model LR yang sudah di-train."""
    protein_encoded = PROTEIN_URINE_MAP.get(
        (protein_urine or "").lower().strip(), 0
    )

    features = _engineer_features(
        systolic=systolic or 120,
        diastolic=diastolic or 80,
        age=age,
        gestational_age_weeks=gestational_age_weeks,
        bmi=bmi,
        has_hypertension_history=has_hypertension_history,
        protein_urine_encoded=protein_encoded,
        has_preeclampsia_history=has_preeclampsia_history,
        heart_rate=heart_rate or 80,
        body_temp=body_temp or 37.0,
        bs=bs or 10.0,
    )

    prob = _pipeline.predict_proba(features)[0][1]
    logger.debug(f"LR predict: prob={prob:.4f}")
    return round(float(prob), 4)


def _predict_heuristic(
    systolic, diastolic, protein_urine, has_preeclampsia_history,
) -> float:
    """Fallback heuristik jika model .pkl belum tersedia."""
    prob = 0.0

    if systolic is not None:
        if systolic >= 160:
            prob += 0.40
        elif systolic >= 140:
            prob += 0.20
        elif systolic >= 130:
            prob += 0.05

    if diastolic is not None:
        if diastolic >= 110:
            prob += 0.30
        elif diastolic >= 90:
            prob += 0.15
        elif diastolic >= 80:
            prob += 0.03

    protein = (protein_urine or "").lower().strip()
    if protein in ("positif_kuat", "+3", "+4"):
        prob += 0.25
    elif protein in ("positif", "+1", "+2"):
        prob += 0.10

    if has_preeclampsia_history:
        prob += 0.15

    return round(min(max(prob, 0.0), 1.0), 4)
