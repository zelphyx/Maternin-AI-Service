"""
MaternIn AI Service — Visit Brief LLM Agent
=============================================
Merangkum riwayat ANC + risk_assessments + postpartum_logs
menjadi ringkasan 2-3 kalimat untuk bidan.

Guardrails (PRD Section 4.5):
  - TIDAK BOLEH menambahkan interpretasi klinis di luar data yang diberikan
  - Hindari halusinasi angka/gejala yang tidak ada di payload
"""

from __future__ import annotations

import json
import logging
from typing import Any

from app.core.config import settings

logger = logging.getLogger("maternin.ai.agent.visit_brief")

FALLBACK_BRIEF = "Ringkasan kunjungan saat ini tidak dapat dibuat. Silakan tinjau data riwayat pasien secara manual."

SYSTEM_PROMPT = """Kamu adalah sistem ringkasan medis untuk bidan di Indonesia.

TUGAS:
Buat ringkasan 2-3 kalimat SINGKAT dari data riwayat kehamilan pasien yang diberikan.

ATURAN MUTLAK:
1. HANYA gunakan data yang ada di payload. JANGAN menambahkan angka, gejala, atau interpretasi yang tidak ada.
2. Gunakan bahasa Indonesia formal yang ringkas.
3. Fokus pada: kondisi terkini, red flags yang pernah muncul, dan tren risiko.
4. JANGAN membuat diagnosis atau rekomendasi tindakan."""

USER_PROMPT_TEMPLATE = """Berikut data riwayat pasien (UUID: REDACTED):

Riwayat ANC:
{anc_summary}

Risk Assessments:
{risk_summary}

Postpartum Logs:
{postpartum_summary}

Buatkan ringkasan 2-3 kalimat untuk bidan."""


def _summarize_data(data: list[dict[str, Any]], max_items: int = 5) -> str:
    """Ringkas list dict jadi string readable, limit items."""
    if not data:
        return "Tidak ada data."
    items = data[-max_items:]  # Ambil yang terbaru
    return json.dumps(items, ensure_ascii=False, default=str, indent=2)[:1500]


def _get_llm():
    """Lazy init ChatGroq."""
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.groq_api_key,
            temperature=0.2,
            max_tokens=256,
            timeout=10,
            max_retries=2,
        )
    except Exception:
        return None


async def generate_visit_brief(
    anc_history: list[dict[str, Any]],
    risk_assessments: list[dict[str, Any]],
    postpartum_logs: list[dict[str, Any]],
    pregnancy_profile_id: str,
) -> str:
    """Generate ringkasan kunjungan 2-3 kalimat."""
    logger.info(
        f"Visit brief request: profile={pregnancy_profile_id}, "
        f"anc={len(anc_history)}, risk={len(risk_assessments)}, pp={len(postpartum_logs)}"
    )

    llm = _get_llm()
    if llm is None:
        return _generate_fallback_brief(anc_history, risk_assessments, postpartum_logs)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        user_msg = USER_PROMPT_TEMPLATE.format(
            anc_summary=_summarize_data(anc_history),
            risk_summary=_summarize_data(risk_assessments),
            postpartum_summary=_summarize_data(postpartum_logs),
        )

        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_msg),
        ])

        brief = response.content.strip()
        if brief:
            logger.info(f"Visit brief generated ({len(brief)} chars)")
            return brief

    except Exception as exc:
        logger.error(f"Visit brief LLM error: {type(exc).__name__}: {exc}")

    return _generate_fallback_brief(anc_history, risk_assessments, postpartum_logs)


def _generate_fallback_brief(
    anc_history: list[dict],
    risk_assessments: list[dict],
    postpartum_logs: list[dict],
) -> str:
    """Fallback: rangkum tanpa LLM."""
    parts = []

    if risk_assessments:
        latest = risk_assessments[-1]
        badge = latest.get("risk_badge", "tidak diketahui")
        score = latest.get("aggregate_score", "N/A")
        parts.append(f"Penilaian risiko terakhir: {badge} (skor {score}/100).")

    if anc_history:
        parts.append(f"Tercatat {len(anc_history)} kunjungan ANC.")

    if postpartum_logs:
        parts.append(f"Tersedia {len(postpartum_logs)} log postpartum.")

    if not parts:
        return FALLBACK_BRIEF

    return " ".join(parts)
