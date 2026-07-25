"""
MaternIn AI Service — Triage Analyze Router
=============================================
POST /api/v1/triage/analyze

Endpoint utama P0 — menerima data checkin, menjalankan pipeline triage,
mengirim alert WA darurat jika merah, dan callback hasil ke NestJS.
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Depends

from app.core.auth import get_request_id, verify_internal_token
from app.pipelines.triage_engine import run_triage_pipeline
from app.schemas.nestjs_callback import RiskAssessmentCallback
from app.schemas.triage import TriageAnalyzeRequest, TriageAnalyzeResponse
from app.services.nestjs_client import post_risk_assessment_callback
from app.services.whatsapp_client import build_emergency_message, send_emergency_alert

logger = logging.getLogger("maternin.ai.router.triage")

router = APIRouter(
    prefix="/api/v1/triage",
    tags=["Triage"],
    dependencies=[Depends(verify_internal_token)],
)


@router.post(
    "/analyze",
    response_model=TriageAnalyzeResponse,
    summary="Analisis risiko kehamilan (Triage Engine)",
    description=(
        "Menerima data checkin gejala & ANC, menjalankan pipeline triage "
        "(rule-based → ML → aggregator → narasi), dan mengembalikan "
        "risk_badge, aggregate_score, risk_factors, serta recommendation_text."
    ),
)
async def analyze_triage(
    request: TriageAnalyzeRequest,
    request_id: str | None = Depends(get_request_id),
) -> TriageAnalyzeResponse:
    """
    Pipeline:
      Lapis 1 (Rule-based) → LR Preeklampsia + CV Anemia → XGBoost Aggregator → LLM Narasi
      → Emergency WA Alert (jika merah)
      → Callback ke NestJS /internal/risk-assessments
    """
    logger.info(
        f"[{request_id}] Triage analyze request: "
        f"profile={request.pregnancy_profile_id}, "
        f"checkin={request.symptom_checkin_id}"
    )

    # Konversi latest_anc model ke dict untuk pipeline
    latest_anc_dict = None
    if request.latest_anc:
        latest_anc_dict = request.latest_anc.model_dump(exclude_none=True)

    # ── Jalankan Pipeline ────────────────────────────────────────────
    result = await run_triage_pipeline(
        answers=request.answers,
        conjunctiva_image_url=request.conjunctiva_image_url,
        latest_anc=latest_anc_dict,
        has_preeclampsia_history=request.has_preeclampsia_history,
    )

    # ── Alert WA Darurat (jika merah) ────────────────────────────────
    if result.risk_badge.value == "merah":
        logger.warning(
            f"[{request_id}] 🔴 RISIKO MERAH terdeteksi! "
            f"profile={request.pregnancy_profile_id}"
        )

        # Kirim WA darurat ke bidan — TIDAK menunda response ke NestJS
        wa_message = build_emergency_message(
            risk_factors=result.risk_factors,
            aggregate_score=result.aggregate_score,
            pregnancy_profile_id=request.pregnancy_profile_id,
        )
        alert_status = await send_emergency_alert(
            phone_number=request.bidan_phone or "",
            message=wa_message,
            request_id=request_id,
        )
        result.alert_delivery_status = alert_status
    else:
        result.alert_delivery_status = "not_triggered"

    # ── Callback ke NestJS (fire-and-forget di background) ───────────
    callback_payload = RiskAssessmentCallback(
        pregnancy_profile_id=request.pregnancy_profile_id,
        symptom_checkin_id=request.symptom_checkin_id,
        triage_score=result.triage_score or 0.0,
        anemia_probability=result.anemia_probability,
        preeclampsia_probability=result.preeclampsia_probability,
        aggregate_score=result.aggregate_score,
        risk_badge=result.risk_badge,
        risk_factors=result.risk_factors,
        recommendation_text=result.recommendation_text,
        alert_delivery_status=result.alert_delivery_status,
        anemia_is_mock=result.anemia_is_mock,
    )

    # Background task: kirim callback tanpa menunda response ke NestJS
    asyncio.create_task(
        _safe_callback(callback_payload, request_id)
    )

    logger.info(
        f"[{request_id}] Triage result: badge={result.risk_badge.value}, "
        f"score={result.aggregate_score}, factors={len(result.risk_factors)}, "
        f"wa_alert={result.alert_delivery_status}"
    )

    return result


async def _safe_callback(
    payload: RiskAssessmentCallback,
    request_id: str | None,
) -> None:
    """
    Wrapper untuk fire-and-forget callback.
    Menangkap semua exception agar tidak crash event loop.
    """
    try:
        await post_risk_assessment_callback(payload, request_id)
    except Exception as exc:
        logger.error(
            f"[{request_id}] Unhandled error in NestJS callback: "
            f"{type(exc).__name__}: {exc}"
        )
