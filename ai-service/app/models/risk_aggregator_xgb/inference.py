"""
MaternIn AI Service — Risk Aggregator XGBoost Inference (Lapis 2)
===================================================================
Memuat model XGBoost .pkl saat startup, fallback ke weighted heuristic.
Mengekstrak feature importance untuk explainable risk_factors.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np

from app.schemas.triage import RiskBadge

logger = logging.getLogger("maternin.ai.risk_aggregator")

# ── State model ──────────────────────────────────────────────────────────
_model_bundle = None
_model_loaded = False

ARTIFACT_FILENAME = "risk_aggregator_v1.pkl"


def _artifact_path() -> str:
    """
    Return path ke .pkl risk aggregator XGBoost bundle.
    Resolved dari HF Hub cache (production) atau folder lokal (dev).
    """
    from app.core.artifact_loader import ensure_model_artifacts
    return str(ensure_model_artifacts() / ARTIFACT_FILENAME)

# ── Fallback thresholds (dipakai jika model belum di-train) ──────────────
BADGE_THRESHOLDS = {
    "merah": 65,
    "kuning": 35,
}

COMPONENT_WEIGHTS = {
    "triage_lapis1": 0.45,
    "preeclampsia": 0.35,
    "anemia": 0.20,
}


def load_model(artifact_path: str | None = None) -> None:
    """
    Load model XGBoost bundle (model + label_encoder + feature_columns).
    Dipanggil saat FastAPI startup (lifespan event).
    """
    global _model_bundle, _model_loaded

    path = artifact_path or _artifact_path()
    if not os.path.exists(path):
        logger.warning(
            f"Risk Aggregator model not found at {path}. "
            f"Using fallback heuristic mode. "
            f"Train with: python app/training/risk_aggregator_train.py"
        )
        _model_loaded = False
        return

    try:
        import joblib
        _model_bundle = joblib.load(path)
        _model_loaded = True

        model = _model_bundle["model"]
        features = _model_bundle["feature_columns"]
        labels = list(_model_bundle["label_encoder"].classes_)

        logger.info(
            f"✅ Risk Aggregator XGBoost loaded from {path} "
            f"(features={features}, labels={labels})"
        )
    except Exception as exc:
        logger.error(f"Failed to load risk aggregator model: {exc}")
        _model_loaded = False


def aggregate_risk(
    triage_score: float,
    preeclampsia_prob: float = 0.0,
    anemia_prob: float | None = None,
    is_absolute_red: bool = False,
    # Optional extra features for XGBoost model
    age: float = 25.0,
    gestational_age_weeks: float = 28.0,
    systolic_bp: float = 120.0,
    diastolic_bp: float = 80.0,
    hemoglobin_g_dl: float = 12.0,
) -> dict[str, Any]:
    """
    Agregasi skor dari semua komponen.

    Jika model XGBoost tersedia: inferensi asli + feature importance.
    Jika tidak: fallback ke weighted heuristic.
    """
    # Absolute red flag dari lapis 1 → langsung merah
    if is_absolute_red:
        aggregate = max(triage_score, BADGE_THRESHOLDS["merah"])
        return {
            "aggregate_score": round(min(aggregate, 100.0), 1),
            "risk_badge": RiskBadge.merah,
            "feature_importances": None,
        }

    if _model_loaded and _model_bundle is not None:
        return _predict_with_model(
            triage_score, preeclampsia_prob, anemia_prob,
            age, gestational_age_weeks, systolic_bp, diastolic_bp, hemoglobin_g_dl,
        )
    else:
        return _predict_heuristic(
            triage_score, preeclampsia_prob, anemia_prob,
        )


def _predict_with_model(
    triage_score, preeclampsia_prob, anemia_prob,
    age, gestational_age_weeks, systolic_bp, diastolic_bp, hemoglobin_g_dl,
) -> dict[str, Any]:
    """Inferensi menggunakan model XGBoost yang sudah di-train.

    Model baru dilatih dengan 29 fitur engineered (lihat training script
    risk_aggregator_improved.py). Fitur di-engineer dari input klinis.
    """
    model = _model_bundle["model"]
    label_encoder = _model_bundle["label_encoder"]
    feature_columns = _model_bundle["feature_columns"]

    # Build the 29-feature vector matching training
    feat = _engineer_xgb_features(
        triage_score=triage_score,
        preeclampsia_prob=preeclampsia_prob or 0.0,
        anemia_prob=anemia_prob or 0.0,
        age=age,
        gestational_age_weeks=gestational_age_weeks,
        systolic_bp=systolic_bp,
        diastolic_bp=diastolic_bp,
        hemoglobin_g_dl=hemoglobin_g_dl,
    )

    # Predict class + probability
    pred_class = model.predict(feat)[0]
    pred_proba = model.predict_proba(feat)[0]

    badge_name = label_encoder.inverse_transform([pred_class])[0]
    badge = RiskBadge(badge_name)

    # Aggregate score from probabilities (weighted)
    aggregate = (pred_proba[0] * 10 + pred_proba[1] * 50 + pred_proba[2] * 100)
    aggregate = round(min(max(aggregate, 0.0), 100.0), 1)

    # Feature importances (not available for VotingClassifier ensemble;
    # fall back to first underlying estimator's importances if accessible)
    importances: list | None = None
    try:
        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_.tolist()
        elif hasattr(model, "estimators_") and len(model.estimators_) > 0:
            # Average importances across estimators that have them (XGB does, RF does)
            imp_arrays = [
                est.feature_importances_ for est in model.estimators_
                if hasattr(est, "feature_importances_")
            ]
            if imp_arrays:
                importances = np.mean(imp_arrays, axis=0).tolist()
    except Exception:
        importances = None

    logger.debug(
        f"XGBoost predict: badge={badge_name}, score={aggregate}, "
        f"proba={pred_proba.tolist()}"
    )

    return {
        "aggregate_score": aggregate,
        "risk_badge": badge,
        "feature_importances": importances,
    }


def _engineer_xgb_features(
    triage_score: float,
    preeclampsia_prob: float,
    anemia_prob: float,
    age: float,
    gestational_age_weeks: float,
    systolic_bp: float,
    diastolic_bp: float,
    hemoglobin_g_dl: float,
) -> np.ndarray:
    """Build 29-feature vector for XGB risk aggregator.

    Mirrors `_engineer_features()` in preeclampsia_lr inference, so both
    models use the same engineered features.
    """
    # Reasonable defaults for vitals when not provided
    heart_rate = 80.0
    body_temp = 37.0
    bs = 10.0
    protein_urine_encoded = 0
    has_preeclampsia_history = 0
    has_hypertension_history = 1 if (systolic_bp >= 140 or diastolic_bp >= 90) else 0
    bmi = 25.0

    engineered = {
        "systolic_bp": systolic_bp,
        "diastolic_bp": diastolic_bp,
        "age": age,
        "gestational_age_weeks": gestational_age_weeks,
        "bmi": bmi,
        "has_hypertension_history": has_hypertension_history,
        "protein_urine_encoded": protein_urine_encoded,
        "has_preeclampsia_history": has_preeclampsia_history,
        "systolic_diastolic_product": systolic_bp * diastolic_bp,
        "pulse_pressure": systolic_bp - diastolic_bp,
        "mean_arterial_pressure": diastolic_bp + (systolic_bp - diastolic_bp) / 3,
        "bp_severity": (systolic_bp - 120) / 30 + (diastolic_bp - 80) / 20,
        "age_bp_interaction": age * systolic_bp / 100,
        "is_extreme_age": int((age < 18) or (age > 35)),
        "is_high_age": int(age >= 35),
        "is_young": int(age < 20),
        "heart_rate_abnormal": int((heart_rate < 60) or (heart_rate > 100)),
        "heart_rate_severity": abs(heart_rate - 80) / 20,
        "fever": int(body_temp > 38.0),
        "hypothermia": int(body_temp < 36.0),
        "bs_abnormal": int((bs < 7) or (bs > 14)),
        "bs_severity": abs(bs - 10) / 5,
        "severe_hypertension": int((systolic_bp >= 160) or (diastolic_bp >= 110)),
        "moderate_hypertension": int(
            ((systolic_bp >= 140) and (systolic_bp < 160)) or
            ((diastolic_bp >= 90) and (diastolic_bp < 110))
        ),
        "abnormal_vitals_count": (
            int((heart_rate < 60) or (heart_rate > 100))
            + int(body_temp > 38.0)
            + int(body_temp < 36.0)
            + int((bs < 7) or (bs > 14))
        ),
        "age_bp_severity": ((systolic_bp - 120) / 30 + (diastolic_bp - 80) / 20) * int(age >= 35),
        "multi_system_abnormality": int(
            ((systolic_bp >= 140) and (systolic_bp < 160)) or
            ((diastolic_bp >= 90) and (diastolic_bp < 110))
        ) * (
            int((heart_rate < 60) or (heart_rate > 100))
            + int(body_temp > 38.0)
            + int(body_temp < 36.0)
            + int((bs < 7) or (bs > 14))
        ),
        "systolic_squared": systolic_bp ** 2 / 1000,
        "diastolic_squared": diastolic_bp ** 2 / 1000,
    }
    # Build vector in the order expected by the model
    return np.array([[engineered[col] for col in [
        "systolic_bp", "diastolic_bp", "age", "gestational_age_weeks", "bmi",
        "has_hypertension_history", "protein_urine_encoded", "has_preeclampsia_history",
        "systolic_diastolic_product", "pulse_pressure", "mean_arterial_pressure", "bp_severity",
        "age_bp_interaction", "is_extreme_age", "is_high_age", "is_young",
        "heart_rate_abnormal", "heart_rate_severity", "fever", "hypothermia", "bs_abnormal", "bs_severity",
        "severe_hypertension", "moderate_hypertension", "abnormal_vitals_count",
        "age_bp_severity", "multi_system_abnormality",
        "systolic_squared", "diastolic_squared",
    ]]])


def _predict_heuristic(
    triage_score, preeclampsia_prob, anemia_prob,
) -> dict[str, Any]:
    """Fallback heuristik jika model belum tersedia."""
    preeclampsia_scaled = (preeclampsia_prob or 0.0) * 100.0
    anemia_scaled = (anemia_prob or 0.0) * 100.0

    if anemia_prob is not None:
        aggregate = (
            triage_score * COMPONENT_WEIGHTS["triage_lapis1"]
            + preeclampsia_scaled * COMPONENT_WEIGHTS["preeclampsia"]
            + anemia_scaled * COMPONENT_WEIGHTS["anemia"]
        )
    else:
        total_weight = COMPONENT_WEIGHTS["triage_lapis1"] + COMPONENT_WEIGHTS["preeclampsia"]
        aggregate = (
            triage_score * (COMPONENT_WEIGHTS["triage_lapis1"] / total_weight)
            + preeclampsia_scaled * (COMPONENT_WEIGHTS["preeclampsia"] / total_weight)
        )

    aggregate = round(min(max(aggregate, 0.0), 100.0), 1)

    if aggregate >= BADGE_THRESHOLDS["merah"]:
        badge = RiskBadge.merah
    elif aggregate >= BADGE_THRESHOLDS["kuning"]:
        badge = RiskBadge.kuning
    else:
        badge = RiskBadge.hijau

    return {
        "aggregate_score": aggregate,
        "risk_badge": badge,
        "feature_importances": None,
    }
