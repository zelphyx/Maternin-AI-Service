"""
MaternIn AI Service — Chat Router
====================================
POST /api/v1/chat

Endpoint chatbot edukasi ibu hamil. Setiap jawaban wajib menyertakan
disclaimer medis (disclaimer_included: true).
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.core.auth import get_request_id, verify_internal_token
from app.agents.chatbot_agent import chat_reply
from app.schemas.chat import ChatRequest, ChatResponse

logger = logging.getLogger("maternin.ai.router.chat")

router = APIRouter(
    prefix="/api/v1/chat",
    tags=["Chatbot"],
    dependencies=[Depends(verify_internal_token)],
)


@router.post(
    "",
    response_model=ChatResponse,
    summary="Chatbot edukasi ibu hamil",
    description=(
        "Menerima pesan dari ibu hamil dan mengembalikan jawaban edukasi "
        "kontekstual dengan disclaimer medis. Chatbot tidak boleh membuat "
        "diagnosis atau menebak angka skor risiko."
    ),
)
async def chat_endpoint(
    request: ChatRequest,
    request_id: str | None = Depends(get_request_id),
) -> ChatResponse:
    """
    Proses pesan chat dan kembalikan jawaban edukasi.
    """
    logger.info(
        f"[{request_id}] Chat request: "
        f"profile={request.pregnancy_profile_id}, "
        f"message_len={len(request.message)}"
    )

    result = await chat_reply(
        message=request.message,
        pregnancy_profile_id=request.pregnancy_profile_id,
    )

    logger.info(
        f"[{request_id}] Chat reply: "
        f"len={len(result['reply'])}, "
        f"disclaimer={result['disclaimer_included']}"
    )

    return ChatResponse(
        reply=result["reply"],
        disclaimer_included=result["disclaimer_included"],
    )
