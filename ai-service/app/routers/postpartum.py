"""
MaternIn AI Service — Postpartum Evaluate Router
===================================================
POST /api/v1/postpartum/evaluate (P1)

Evaluasi checklist harian nifas:
- Red flags: perdarahan, infeksi luka, demam, sakit kepala hebat
- Mental health flag: pola mood_flag "sering_sedih" beruntun
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends

from app.core.auth import get_request_id, verify_internal_token
from app.schemas.postpartum import (
    BleedingLevel,
    MoodFlag,
    PostpartumEvaluateRequest,
    PostpartumEvaluateResponse,
    PostpartumLogEntry,
    WoundCondition,
)

logger = logging.getLogger("maternin.ai.router.postpartum")

router = APIRouter(
    prefix="/api/v1/postpartum",
    tags=["Postpartum"],
    dependencies=[Depends(verify_internal_token)],
)


def _evaluate_red_flags(
    logs: list[PostpartumLogEntry],
    had_preeclampsia_history: bool,
) -> dict[str, Any]:
    """
    Evaluasi red flags nifas dari log harian.

    Red flag criteria (berdasarkan PNPK):
    - Perdarahan banyak / sangat banyak
    - Demam (>38°C)
    - Luka jahitan bau atau bengkak/merah (tanda infeksi)
    - Sakit kepala hebat (terutama jika ada riwayat preeklampsia)
    """
    reasons: list[str] = []
    red_flag = False

    for log in logs:
        day = f"Hari ke-{log.day_number}"

        # Perdarahan banyak
        if log.bleeding_level == BleedingLevel.sangat_banyak:
            reasons.append(f"{day}: Perdarahan sangat banyak (darurat)")
            red_flag = True
        elif log.bleeding_level == BleedingLevel.banyak:
            reasons.append(f"{day}: Perdarahan banyak")
            red_flag = True

        # Demam
        if log.fever:
            reasons.append(f"{day}: Demam")
            red_flag = True

        # Infeksi luka
        if log.wound_condition == WoundCondition.bengkak_merah:
            reasons.append(f"{day}: Luka bengkak/merah (tanda infeksi)")
            red_flag = True
        elif log.wound_condition == WoundCondition.bau:
            reasons.append(f"{day}: Luka berbau (tanda infeksi)")
            red_flag = True

        # Sakit kepala hebat
        if log.headache_severe:
            reasons.append(f"{day}: Sakit kepala hebat")
            red_flag = True

            # Lebih serius jika ada riwayat preeklampsia
            if had_preeclampsia_history:
                reasons.append(
                    f"{day}: ⚠️ Sakit kepala hebat + riwayat preeklampsia — "
                    "risiko eklamsia pascamelahirkan"
                )

    return {
        "red_flag_triggered": red_flag,
        "reason": " + ".join(reasons) if reasons else "",
    }


def _evaluate_mental_health(logs: list[PostpartumLogEntry]) -> bool:
    """
    Evaluasi pola mood_flag untuk deteksi potensi baby blues.

    Criteria:
    - 3+ hari berturut-turut mood "sering_sedih"
    - ATAU 5+ hari (tidak berturut-turut) mood "sering_sedih" dari total log
    """
    sorted_logs = sorted(logs, key=lambda x: x.day_number)

    # Check consecutive "sering_sedih"
    consecutive = 0
    max_consecutive = 0
    total_sering_sedih = 0

    for log in sorted_logs:
        if log.mood_flag == MoodFlag.sering_sedih:
            consecutive += 1
            total_sering_sedih += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0

    # Trigger if 3+ consecutive or 5+ total
    if max_consecutive >= 3:
        return True
    if total_sering_sedih >= 5:
        return True

    return False


@router.post(
    "/evaluate",
    response_model=PostpartumEvaluateResponse,
    summary="Evaluasi nifas harian",
    description=(
        "Evaluasi checklist harian masa nifas untuk mendeteksi red flags "
        "(perdarahan, infeksi, demam, sakit kepala) dan indikasi mental health."
    ),
)
async def postpartum_evaluate(
    request: PostpartumEvaluateRequest,
    request_id: str | None = Depends(get_request_id),
) -> PostpartumEvaluateResponse:
    """Evaluasi log postpartum dan kembalikan red flag + mental health flag."""
    logger.info(
        f"[{request_id}] Postpartum evaluate: "
        f"profile={request.pregnancy_profile_id}, "
        f"logs={len(request.logs)}, "
        f"preeclampsia_history={request.had_preeclampsia_history}"
    )

    # Evaluate red flags
    red_flag_result = _evaluate_red_flags(
        logs=request.logs,
        had_preeclampsia_history=request.had_preeclampsia_history,
    )

    # Evaluate mental health
    mental_health_flag = _evaluate_mental_health(request.logs)

    response = PostpartumEvaluateResponse(
        red_flag_triggered=red_flag_result["red_flag_triggered"],
        reason=red_flag_result["reason"],
        mental_health_flag=mental_health_flag,
    )

    logger.info(
        f"[{request_id}] Postpartum result: "
        f"red_flag={response.red_flag_triggered}, "
        f"mental_health={response.mental_health_flag}"
    )

    # Fire-and-forget callback to NestJS
    if response.red_flag_triggered or response.mental_health_flag:
        logger.warning(
            f"[{request_id}] ⚠️ Postpartum flag(s) triggered! "
            f"profile={request.pregnancy_profile_id}"
        )
        import asyncio
        from app.services.nestjs_client import post_postpartum_flag_callback
        from app.schemas.nestjs_callback import PostpartumFlagCallback

        callback_payload = PostpartumFlagCallback(
            pregnancy_profile_id=request.pregnancy_profile_id,
            red_flag_triggered=response.red_flag_triggered,
            reason=response.reason,
            mental_health_flag=response.mental_health_flag,
        )

        asyncio.create_task(
            post_postpartum_flag_callback(
                payload=callback_payload,
                request_id=request_id,
            )
        )

    return response
