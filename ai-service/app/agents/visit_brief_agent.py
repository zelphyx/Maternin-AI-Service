"""
MaternIn AI Service — Visit Brief Agent
========================================
Delega ke llm_client.py untuk HTTP request (bypass langchain).
"""

from __future__ import annotations

from typing import Any

from app.agents.llm_client import generate_visit_brief as _brief


async def generate_visit_brief(
    anc_history: list[dict[str, Any]],
    risk_assessments: list[dict[str, Any]],
    postpartum_logs: list[dict[str, Any]],
    pregnancy_profile_id: str,
) -> str:
    return await _brief(
        anc_history=anc_history,
        risk_assessments=risk_assessments,
        postpartum_logs=postpartum_logs,
        pregnancy_profile_id=pregnancy_profile_id,
    )
