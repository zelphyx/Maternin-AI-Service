"""
MaternIn AI Service — Chatbot Agent
====================================
Delega ke llm_client.py untuk HTTP request (bypass langchain).
"""

from __future__ import annotations

from typing import Any

from app.agents.llm_client import chat_reply as _chat
from app.agents.llm_client import _parse_fallback as _fallback_chat


async def chat_reply(
    message: str,
    pregnancy_profile_id: str,
    grounding_context: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    return await _chat(
        message=message,
        pregnancy_profile_id=pregnancy_profile_id,
        grounding_context=grounding_context,
        **kwargs,
    )
