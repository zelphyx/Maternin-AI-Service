"""
MaternIn AI Service — Triage Pipeline Orchestrator
====================================================
Menghubungkan seluruh tahapan pipeline analisis triage:

  Lapis 1: Rule-based triage (deterministik, PNPK)
       ↓
  Lapis Inferensi (paralel):
    - Logistic Regression (preeklampsia)
    - Computer Vision (anemia konjungtiva)
       ↓
  Lapis 2: XGBoost Aggregator (skor akhir + badge)
       ↓
  Lapis 3: LLM Narasi Rekomendasi (TODO: Task 05)

Output pipeline = TriageAnalyzeResponse lengkap.
"""

from __future__ import annotations

import logging
from typing import Any

from app.models.triage_rules import evaluate_triage
from app.models.preeclampsia_lr.inference import predict_preeclampsia
from app.models.anemia_cv.inference import predict_anemia, is_mock_mode as is_anemia_mock
from app.models.risk_aggregator_xgb.inference import aggregate_risk
from app.agents.recommendation_agent import generate_recommendation
from app.schemas.triage import RiskBadge, TriageAnalyzeResponse

logger = logging.getLogger("maternin.ai.pipeline.triage")


async def run_triage_pipeline(
    answers: dict[str, Any],
    conjunctiva_image_url: str | None = None,
    latest_anc: dict[str, Any] | None = None,
    has_preeclampsia_history: bool = False,
) -> TriageAnalyzeResponse:
    """
    Jalankan pipeline triage lengkap (Lapis 1 → Inferensi → Lapis 2 → Lapis 3).

    Args:
        answers: Jawaban kuesioner adaptif dari pasien.
        conjunctiva_image_url: URL gambar konjungtiva (opsional).
        latest_anc: Data ANC terakhir (tensi, protein urine, dll).
        has_preeclampsia_history: Riwayat preeklampsia.

    Returns:
        TriageAnalyzeResponse lengkap.
    """
    anc = latest_anc or {}

    # ── Lapis 1: Rule-Based Triage ───────────────────────────────────
    logger.info("Pipeline: running Lapis 1 (rule-based triage)...")
    triage_result = evaluate_triage(
        answers=answers,
        latest_anc=anc,
        has_preeclampsia_history=has_preeclampsia_history,
    )
    logger.info(
        f"  Lapis 1 result: score={triage_result.triage_score}, "
        f"flags={len(triage_result.red_flags)}, absolute_red={triage_result.is_absolute_red}"
    )

    # ── Lapis Inferensi (paralel): Preeklampsia + Anemia ─────────────
    logger.info("Pipeline: running Lapis Inferensi (LR + CV)...")

    # Preeklampsia (Logistic Regression)
    preeclampsia_prob = predict_preeclampsia(
        systolic=anc.get("systolic"),
        diastolic=anc.get("diastolic"),
        protein_urine=anc.get("protein_urine"),
        has_preeclampsia_history=has_preeclampsia_history,
    )
    logger.info(f"  Preeclampsia LR: prob={preeclampsia_prob}")

    # Anemia (Computer Vision) — async karena perlu download gambar
    anemia_prob: float | None = None
    anemia_is_mock = False
    try:
        anemia_prob = await predict_anemia(image_url=conjunctiva_image_url)
        anemia_is_mock = is_anemia_mock()
        logger.info(f"  Anemia CV: prob={anemia_prob}, is_mock={anemia_is_mock}")
    except Exception as e:
        # Pipeline tetap lanjut meski CV gagal — catat saja
        anemia_is_mock = True
        logger.warning(f"  Anemia CV failed (pipeline continues): {e}")

    # ── Lapis 2: XGBoost Aggregator ──────────────────────────────────
    logger.info("Pipeline: running Lapis 2 (risk aggregator)...")
    aggregation = aggregate_risk(
        triage_score=triage_result.triage_score,
        preeclampsia_prob=preeclampsia_prob,
        anemia_prob=anemia_prob,
        is_absolute_red=triage_result.is_absolute_red,
    )
    logger.info(
        f"  Aggregator result: score={aggregation['aggregate_score']}, "
        f"badge={aggregation['risk_badge'].value}"
    )

    # ── Lapis 3: Narasi Rekomendasi (LLM) ────────────────────────────
    logger.info("Pipeline: running Lapis 3 (LLM recommendation)...")
    recommendation_text = await generate_recommendation(
        risk_badge=aggregation["risk_badge"].value,
        aggregate_score=aggregation["aggregate_score"],
        risk_factors=triage_result.red_flags,
    )
    logger.info(f"  LLM recommendation: {len(recommendation_text)} chars")

    # ── Compose Response ─────────────────────────────────────────────
    return TriageAnalyzeResponse(
        risk_badge=aggregation["risk_badge"],
        aggregate_score=aggregation["aggregate_score"],
        risk_factors=triage_result.red_flags,
        recommendation_text=recommendation_text,
        triage_score=triage_result.triage_score,
        anemia_probability=anemia_prob,
        preeclampsia_probability=preeclampsia_prob,
        alert_delivery_status=None,  # Diisi oleh router setelah kirim WA
        anemia_is_mock=anemia_is_mock,
    )


def _generate_mock_recommendation(
    risk_badge: RiskBadge,
    risk_factors: list[str],
) -> str:
    """
    Generasi narasi rekomendasi mock (Lapis 3).
    TODO [Task 05]: Ganti dengan LangChain + GROQ/Qwen.
    """
    factors_text = ", ".join(risk_factors) if risk_factors else "Tidak ada faktor risiko signifikan"

    if risk_badge == RiskBadge.merah:
        return (
            f"⚠️ PERHATIAN — Risiko Tinggi. "
            f"Segera hubungi bidan atau kunjungi IGD terdekat. "
            f"Faktor risiko terdeteksi: {factors_text}. "
            f"Jangan menunda — kondisi ini memerlukan penanganan medis segera."
        )
    elif risk_badge == RiskBadge.kuning:
        return (
            f"⚡ Perhatian — Risiko Sedang. "
            f"Jadwalkan kunjungan ke bidan dalam waktu dekat untuk pemeriksaan lanjutan. "
            f"Faktor yang perlu dipantau: {factors_text}. "
            f"Istirahat cukup dan pantau gejala secara rutin."
        )
    else:
        return (
            f"✅ Kondisi Baik — Risiko Rendah. "
            f"Lanjutkan pemeriksaan rutin sesuai jadwal ANC. "
            f"Jaga pola makan bergizi dan istirahat cukup. "
            f"Hubungi bidan jika muncul keluhan baru."
        )
