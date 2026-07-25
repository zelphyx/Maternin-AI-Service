"""
MaternIn AI Service — Visit Brief Router
==========================================
POST /api/v1/visit-brief/generate (P2)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.core.auth import get_request_id, verify_internal_token
from app.agents.visit_brief_agent import generate_visit_brief
from app.schemas.visit_brief import VisitBriefGenerateRequest, VisitBriefGenerateResponse

logger = logging.getLogger("maternin.ai.router.visit_brief")

router = APIRouter(
    prefix="/api/v1/visit-brief",
    tags=["Visit Brief"],
    dependencies=[Depends(verify_internal_token)],
)


@router.post(
    "/generate",
    response_model=VisitBriefGenerateResponse,
    summary="Generate ringkasan kunjungan bidan",
    description=(
        "Merangkum riwayat ANC, risk assessments, dan postpartum logs "
        "menjadi ringkasan 2-3 kalimat untuk persiapan kunjungan bidan."
    ),
)
async def visit_brief_generate(
    request: VisitBriefGenerateRequest,
    request_id: str | None = Depends(get_request_id),
) -> VisitBriefGenerateResponse:
    """Generate visit brief."""
    logger.info(
        f"[{request_id}] Visit brief: "
        f"profile={request.pregnancy_profile_id}, "
        f"anc={len(request.anc_history)}, "
        f"risk={len(request.risk_assessments)}, "
        f"pp={len(request.postpartum_logs)}"
    )

    brief = await generate_visit_brief(
        anc_history=request.anc_history,
        risk_assessments=request.risk_assessments,
        postpartum_logs=request.postpartum_logs,
        pregnancy_profile_id=request.pregnancy_profile_id,
    )

    logger.info(f"[{request_id}] Visit brief result: {len(brief)} chars")

    return VisitBriefGenerateResponse(brief_text=brief)
