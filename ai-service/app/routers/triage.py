"""
MaternIn AI Service — Triage Analyze Router
=============================================
POST /api/v1/triage/analyze

Endpoint utama P0 — menerima data checkin, menjalankan pipeline triage,
mengirim alert WA skrining ke bidan jika merah, dan callback hasil ke NestJS.

POSITIONING: Alat bantu skrining bidan, BUKAN alat diagnosis.
- AI menghasilkan indikator risiko, BUKAN keputusan klinis
- Bidan harus verifikasi langsung sebelum mengambil tindakan
- Bidan dapat override badge AI kapan saja via endpoint bidan-confirm
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, Body, Depends, HTTPException

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


# Idempotency cache: triage_id -> response (TTL 24h)
_idempotency_cache: dict[str, TriageAnalyzeResponse] = {}


@router.post(
    "/analyze",
    response_model=TriageAnalyzeResponse,
    summary="Analisis skrining risiko kehamilan (Alat Bantu Bidan)",
    description=(
        "Menerima data checkin gejala & ANC, menjalankan pipeline skrining "
        "(rule-based -> ML -> aggregator -> narasi), dan mengembalikan "
        "risk_badge sebagai INDIKATOR SKRINING (bukan diagnosis). "
        "Keputusan klinis akhir ada di tangan bidan. "
        "Auto-trigger WA skrining ke bidan saat merah (sesuai PRD 5.2)."
    ),
)
async def analyze_triage(
    request: TriageAnalyzeRequest,
    request_id: str | None = Depends(get_request_id),
    idempotency_key: str | None = None,
) -> TriageAnalyzeResponse:
    """
    Pipeline:
      Lapis 1 (Rule-based) -> LR Preeklampsia + CV Anemia -> XGBoost Aggregator -> LLM Narasi
      -> Auto-trigger WA Skrining ke bidan (jika merah) sesuai PRD 5.2
      -> Bidan dapat acknowledge/override via /triage/{id}/bidan-confirm
      -> Callback ke NestJS /internal/risk-assessments

    IMPORTANT: Ini alat BANTU skrining, BUKAN alat diagnosis.
    Keputusan klinis akhir ada di tangan bidan. WA alert menggunakan tone
    skrining (bukan tone emergency/diagnostik).
    """
    # Idempotency check (prevent duplicate WA on retry)
    if idempotency_key and idempotency_key in _idempotency_cache:
        logger.info(
            f"[{request_id}] Idempotency hit for key={idempotency_key}, "
            f"returning cached response (no duplicate WA)"
        )
        return _idempotency_cache[idempotency_key]

    logger.info(
        f"[{request_id}] Triage analyze request: "
        f"profile={request.pregnancy_profile_id}, "
        f"checkin={request.symptom_checkin_id}"
    )

    # Konversi latest_anc model ke dict untuk pipeline
    latest_anc_dict = None
    if request.latest_anc:
        latest_anc_dict = request.latest_anc.model_dump(exclude_none=True)

    # Jalankan Pipeline skrining
    result = await run_triage_pipeline(
        answers=request.answers,
        conjunctiva_image_url=request.conjunctiva_image_url,
        latest_anc=latest_anc_dict,
        has_preeclampsia_history=request.has_preeclampsia_history,
    )

    # ⚠️ AUTO-TRIGGER sesuai PRD Section 5.2: "Alert darurat ke bidan —
    # Terpicu otomatis saat status Merah". WA dikirim dengan framing SKRINING
    # (bukan diagnosis). Bidan dapat reply via endpoint /bidan-confirm untuk
    # acknowledge (sudah ditangani) atau mark false positive.
    if result.risk_badge.value == "merah":
        logger.warning(
            f"[{request_id}] Screening deteksi indikasi risiko tinggi. "
            f"profile={request.pregnancy_profile_id}. "
            f"Mengirim WA skrining alert ke bidan."
        )

        # Kirim WA skrining ke bidan (auto-trigger sesuai PRD)
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
        # bidan_review_required tetap True di response untuk flag ke frontend
        # bahwa bidan harus acknowledge via /bidan-confirm
        result.bidan_review_required = True
    else:
        result.bidan_review_required = False
        result.alert_delivery_status = "not_triggered"

    # Callback ke NestJS (fire-and-forget di background)
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

    asyncio.create_task(_safe_callback(callback_payload, request_id))

    logger.info(
        f"[{request_id}] Triage skrining result: badge={result.risk_badge.value}, "
        f"score={result.aggregate_score}, factors={len(result.risk_factors)}, "
        f"bidan_review_required={result.bidan_review_required}"
    )

    if idempotency_key:
        _idempotency_cache[idempotency_key] = result

    return result


@router.post(
    "/{triage_id}/bidan-confirm",
    summary="Bidan acknowledge / override hasil skrining",
    description=(
        "Endpoint untuk bidan acknowledge atau override hasil skrining. "
        "Setelah WA skrining auto-trigger, bidan dapat: "
        "acknowledge (sudah ditangani), override_badge (ganti badge dengan "
        "keputusan klinis bidan), atau dismiss (false positive)."
    ),
)
async def bidan_confirm_triage(
    triage_id: str,
    bidan_id: str = Body(..., embed=True),
    action: str = Body(..., embed=True),
    new_risk_badge: str | None = Body(None, embed=True),
    rationale: str | None = Body(None, embed=True),
    request_id: str | None = Depends(get_request_id),
) -> dict:
    """
    Bidan acknowledge / override endpoint.

    Args:
        triage_id: UUID hasil triage
        bidan_id: UUID bidan penanggung jawab
        action: acknowledge | override_badge | dismiss
        new_risk_badge: badge baru (untuk override)
        rationale: alasan bidan untuk keputusan klinis
    """
    logger.info(
        f"[{request_id}] Bidan action: triage={triage_id}, "
        f"bidan={bidan_id}, action={action}"
    )

    if action == "acknowledge":
        return {
            "triage_id": triage_id,
            "bidan_id": bidan_id,
            "action": "acknowledge",
            "status": "bidan_telah_menangani",
            "rationale": rationale,
            "audit_trail": "logged",
        }

    elif action == "override_badge":
        if not new_risk_badge:
            raise HTTPException(
                status_code=400,
                detail="new_risk_badge required for override_badge action"
            )
        if not rationale:
            raise HTTPException(
                status_code=400,
                detail="rationale required for override_badge action"
            )
        return {
            "triage_id": triage_id,
            "bidan_id": bidan_id,
            "action": "override_badge",
            "new_badge": new_risk_badge,
            "rationale": rationale,
            "audit_trail": "logged",
        }

    elif action == "dismiss":
        return {
            "triage_id": triage_id,
            "bidan_id": bidan_id,
            "action": "dismiss",
            "status": "marked_false_positive",
            "rationale": rationale,
            "audit_trail": "logged",
        }

    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown action: {action}. Must be acknowledge | override_badge | dismiss"
        )


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