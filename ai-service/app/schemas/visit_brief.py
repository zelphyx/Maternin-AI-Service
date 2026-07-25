"""
Pydantic Schemas — Visit Brief Generate Endpoint
==================================================
Kontrak request/response untuk POST /api/v1/visit-brief/generate (P2).
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class VisitBriefGenerateRequest(BaseModel):
    """Request body untuk POST /api/v1/visit-brief/generate."""

    pregnancy_profile_id: str = Field(
        ..., description="UUID profil kehamilan pasien"
    )
    anc_history: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Riwayat ANC records pasien",
    )
    risk_assessments: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Riwayat risk assessments pasien",
    )
    postpartum_logs: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Riwayat log postpartum pasien",
    )


class VisitBriefGenerateResponse(BaseModel):
    """Response body dari POST /api/v1/visit-brief/generate."""

    brief_text: str = Field(
        ..., description="Ringkasan 2-3 kalimat riwayat + red flag terbaru pasien"
    )
