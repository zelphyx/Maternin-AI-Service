"""
MaternIn AI Service — Nutrition Parser Agent
==========================================
Delega ke llm_client.py untuk HTTP request (bypass langchain).
"""

from __future__ import annotations

from app.agents.llm_client import parse_nutrition as _parse
from app.agents.llm_client import _parse_fallback as _fallback


async def parse_nutrition(
    raw_message: str,
    pregnancy_profile_id: str,
) -> dict[str, Any]:
    return await _parse(
        raw_message=raw_message,
        pregnancy_profile_id=pregnancy_profile_id,
    )
