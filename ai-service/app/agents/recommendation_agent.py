"""
MaternIn AI Service — Recommendation Narrative Agent (Triage Lapis 3)
====================================================================
Delega ke llm_client.py untuk HTTP request (bypass langchain).
"""

from __future__ import annotations

from app.agents.llm_client import generate_recommendation as _generate
from typing import Any


async def generate_recommendation(
    risk_badge: str,
    aggregate_score: float,
    risk_factors: list[str],
    gestational_age_weeks: float | None = None,
    **kwargs: Any,
) -> str:
    return await _generate(
        risk_badge=risk_badge,
        aggregate_score=aggregate_score,
        risk_factors=risk_factors,
        gestational_age_weeks=gestational_age_weeks,
        **kwargs,
    )
