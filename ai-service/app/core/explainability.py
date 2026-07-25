"""
MaternIn AI Service — Explainability Utility
==============================================
Ekstrak feature importance dari XGBoost dan petakan ke faktor risiko klinis
yang transparan dan dapat ditelusuri (explainable AI).

Digunakan di pipeline triage Lapis 2 untuk mengisi `risk_factors` di response.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("maternin.ai.explainability")

# Mapping dari feature name ke template penjelasan klinis
FEATURE_CLINICAL_MAPPING = {
    "triage_lapis1_score": "Skor triage dasar ({value:.0f}/100)",
    "preeclampsia_risk_prob": "Probabilitas preeklampsia ({value:.0%})",
    "anemia_risk_prob": "Probabilitas anemia ({value:.0%})",
    "age": "Usia ibu {value:.0f} tahun",
    "gestational_age_weeks": "Usia kehamilan {value:.0f} minggu",
    "systolic_bp": "Tekanan sistolik {value:.0f} mmHg",
    "diastolic_bp": "Tekanan diastolik {value:.0f} mmHg",
    "hemoglobin_g_dl": "Kadar Hb {value:.1f} g/dL",
}

# Threshold per fitur untuk menentukan apakah nilai "abnormal"
ABNORMAL_THRESHOLDS = {
    "triage_lapis1_score": lambda v: v >= 30,
    "preeclampsia_risk_prob": lambda v: v >= 0.3,
    "anemia_risk_prob": lambda v: v >= 0.3,
    "age": lambda v: v < 18 or v > 35,
    "gestational_age_weeks": lambda v: v < 8 or v > 40,
    "systolic_bp": lambda v: v >= 140,
    "diastolic_bp": lambda v: v >= 90,
    "hemoglobin_g_dl": lambda v: v < 11.0,
}


def extract_explainable_factors(
    feature_names: list[str],
    feature_values: list[float] | dict[str, float],
    feature_importances: list[float] | None = None,
    top_k: int = 5,
) -> list[str]:
    """
    Ekstrak faktor risiko klinis yang transparan dari fitur model.

    Args:
        feature_names: Nama-nama fitur model.
        feature_values: Nilai fitur pasien saat ini (list atau dict).
        feature_importances: Bobot importance dari XGBoost (opsional).
        top_k: Jumlah maksimal faktor yang ditampilkan.

    Returns:
        List string faktor risiko yang bisa dibaca manusia.
    """
    # Normalisasi input ke dict
    if isinstance(feature_values, (list, tuple)):
        values_dict = dict(zip(feature_names, feature_values))
    else:
        values_dict = feature_values

    # Buat list (feature_name, value, importance)
    factors = []
    for i, name in enumerate(feature_names):
        value = values_dict.get(name, 0.0)
        importance = feature_importances[i] if feature_importances else 1.0

        # Hanya tampilkan fitur dengan nilai abnormal
        threshold_fn = ABNORMAL_THRESHOLDS.get(name)
        if threshold_fn and threshold_fn(value):
            template = FEATURE_CLINICAL_MAPPING.get(name, f"{name}: {{value}}")
            explanation = template.format(value=value)
            factors.append((explanation, importance))

    # Sort by importance (descending) dan ambil top_k
    factors.sort(key=lambda x: -x[1])
    return [f[0] for f in factors[:top_k]]


def merge_risk_factors(
    rule_based_flags: list[str],
    ml_explainable_flags: list[str],
) -> list[str]:
    """
    Gabungkan faktor risiko dari rule-based (Lapis 1) dan ML (Lapis 2).
    Prioritas: rule-based dulu (lebih spesifik), lalu tambahan dari ML.
    Deduplikasi berdasarkan kesamaan konten.
    """
    seen = set()
    merged = []

    for flag in rule_based_flags:
        key = flag.lower().strip()
        if key not in seen:
            seen.add(key)
            merged.append(flag)

    for flag in ml_explainable_flags:
        key = flag.lower().strip()
        if key not in seen:
            seen.add(key)
            merged.append(flag)

    return merged
