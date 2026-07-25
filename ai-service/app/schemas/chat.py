"""
Pydantic Schemas — Chat Endpoint
==================================
Kontrak request/response untuk POST /api/v1/chat (P0).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request body untuk POST /api/v1/chat."""

    pregnancy_profile_id: str = Field(
        ..., description="UUID profil kehamilan pasien"
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Pesan dari ibu hamil",
    )


class ChatResponse(BaseModel):
    """Response body dari POST /api/v1/chat."""

    reply: str = Field(
        ..., description="Jawaban chatbot edukasi"
    )
    disclaimer_included: bool = Field(
        default=True,
        description="Apakah disclaimer medis disertakan di dalam reply",
    )
