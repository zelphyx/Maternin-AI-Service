"""
MaternIn AI Service — Chatbot Agent (Asisten Edukasi Ibu Hamil)
================================================================
LangChain + GROQ API untuk endpoint POST /api/v1/chat.

Guardrails (PRD Section 4.3):
  - Setiap jawaban tentang gejala/kondisi medis wajib menyertakan disclaimer
  - Chatbot DILARANG menebak angka risk score sendiri
  - Jika user tanya soal skor risiko, wajib merujuk pada data resmi
  - Jawaban bersifat edukasi, BUKAN pengganti tenaga medis
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from app.core.config import settings

logger = logging.getLogger("maternin.ai.agent.chatbot")

# ── Fallback response ────────────────────────────────────────────────────
FALLBACK_REPLY = (
    "Maaf, saat ini saya tidak dapat memproses pertanyaan Anda. "
    "Silakan coba lagi nanti atau konsultasikan langsung dengan bidan Anda.\n\n"
    "⚕️ *Jawaban ini bersifat edukasi dan bukan pengganti saran atau diagnosis medis resmi.*"
)

DISCLAIMER = (
    "\n\n⚕️ *Jawaban ini bersifat edukasi dan bukan pengganti saran atau diagnosis medis resmi. "
    "Untuk penanganan lebih lanjut, konsultasikan dengan bidan atau dokter Anda.*"
)

# ── System prompt ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """Kamu adalah MaternIn, asisten virtual edukasi kesehatan ibu hamil di Indonesia.

IDENTITAS:
- Kamu BUKAN dokter atau bidan. Kamu adalah asisten edukasi.
- Kamu memberikan informasi kesehatan kehamilan berdasarkan pedoman Kemenkes RI dan POGI.

ATURAN MUTLAK:
1. SELALU akhiri jawaban dengan disclaimer: "Jawaban ini bersifat edukasi dan bukan pengganti saran atau diagnosis medis resmi."
2. JANGAN PERNAH membuat diagnosis medis.
3. JANGAN PERNAH memberikan angka skor risiko sendiri. Jika ditanya soal skor risiko, jawab: "Untuk mengetahui skor risiko Anda, silakan gunakan fitur Triage Analyze di aplikasi MaternIn atau konsultasi dengan bidan."
4. JANGAN PERNAH meresepkan obat atau dosis obat.
5. Jika user menanyakan hal di luar topik kehamilan/kesehatan ibu, arahkan kembali dengan sopan.
6. Gunakan bahasa Indonesia yang sederhana, hangat, dan mudah dipahami.
7. Jika user menyebut gejala darurat (perdarahan, kejang, pandangan kabur, sakit kepala hebat), SEGERA anjurkan ke IGD terdekat.

TOPIK YANG BOLEH DIJAWAB:
- Nutrisi ibu hamil dan menyusui
- Tanda-tanda bahaya kehamilan
- Perkembangan janin per trimester
- Persiapan persalinan
- Perawatan pascamelahirkan (nifas)
- Imunisasi ibu hamil
- Olahraga dan aktivitas fisik aman untuk ibu hamil
- Keluhan umum kehamilan (mual, nyeri punggung, dll)
- ASI dan menyusui

KONTEKS GROUNDING:
{grounding_context}"""

# ── Load grounding knowledge base ────────────────────────────────────────
_grounding_knowledge: str | None = None


def _load_grounding_kb() -> str:
    """Load Q&A grounding knowledge base dari dataset.

    Selection strategy: stratified sampling agar kategori sparse (1 item)
    tetap terwakili. Prioritas: 1 item dari setiap kategori sparse + sisanya
    dari kategori besar (diverse sample).
    """
    global _grounding_knowledge
    if _grounding_knowledge is not None:
        return _grounding_knowledge

    kb_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "datasets",
        "buku_kia_kemenkes", "maternal_health_qa_kemenkes_500.json"
    )

    try:
        if os.path.exists(kb_path):
            with open(kb_path, "r", encoding="utf-8") as f:
                qa_data = json.load(f)

            if isinstance(qa_data, list):
                all_items = qa_data
            elif isinstance(qa_data, dict) and "data" in qa_data:
                all_items = qa_data["data"]
            else:
                all_items = []

            # Group by category
            from collections import defaultdict
            by_category = defaultdict(list)
            for item in all_items:
                cat = item.get("category", item.get("kategori", "unknown"))
                q = item.get("question", item.get("pertanyaan", ""))
                a = item.get("answer", item.get("jawaban", ""))
                if q and a:
                    by_category[cat].append({"q": q, "a": a})

            # Stratified selection: max 2 per category, min 1 for sparse categories
            MAX_PER_CATEGORY = 2
            selected = []
            seen = set()

            # First pass: 1 item from every sparse category (1-item categories)
            for cat, items in by_category.items():
                if len(items) <= 2:
                    selected.append(items[0])
                    seen.add(id(items[0]))

            # Second pass: up to MAX_PER_CATEGORY from each category, round-robin
            TOTAL_TARGET = 20
            while len(selected) < TOTAL_TARGET:
                added_this_round = 0
                for cat, items in by_category.items():
                    if len(selected) >= TOTAL_TARGET:
                        break
                    for item in items:
                        if id(item) not in seen and len(selected) < TOTAL_TARGET:
                            selected.append(item)
                            seen.add(id(item))
                            added_this_round += 1
                # Safety: if no progress, break
                if added_this_round == 0:
                    break

            context_parts = [f"Q: {item['q']}\nA: {item['a']}" for item in selected]
            _grounding_knowledge = "\n\n".join(context_parts)
            logger.info(
                f"Loaded {len(context_parts)} Q&A grounding items "
                f"({len(by_category)} categories)"
            )
        else:
            _grounding_knowledge = "Tidak ada knowledge base tersedia."
            logger.warning(f"Grounding KB not found at {kb_path}")

    except Exception as exc:
        logger.warning(f"Failed to load grounding KB: {exc}")
        _grounding_knowledge = "Tidak ada knowledge base tersedia."

    return _grounding_knowledge


def _get_llm():
    """Lazy init LangChain ChatGroq."""
    try:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.groq_api_key,
            temperature=0.4,
            max_tokens=768,
            timeout=15,
            max_retries=2,
        )
    except ImportError:
        logger.warning("langchain-groq not installed")
        return None
    except Exception as exc:
        logger.warning(f"Failed to init ChatGroq: {exc}")
        return None


async def chat_reply(
    message: str,
    pregnancy_profile_id: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """
    Proses pesan dari ibu hamil dan hasilkan jawaban edukasi.

    Args:
        message: Pesan dari user.
        pregnancy_profile_id: UUID profil (untuk logging, TIDAK dikirim ke LLM).

    Returns:
        dict dengan "reply" (str) dan "disclaimer_included" (bool).
    """
    logger.info(
        f"Chat request: profile={pregnancy_profile_id}, "
        f"message_len={len(message)}"
    )

    llm = _get_llm()
    if llm is None:
        return {"reply": FALLBACK_REPLY, "disclaimer_included": True}

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        # Load grounding knowledge
        grounding = _load_grounding_kb()

        system_content = SYSTEM_PROMPT.format(grounding_context=grounding)

        messages = [
            SystemMessage(content=system_content),
            HumanMessage(content=message),
        ]

        response = await llm.ainvoke(messages)
        reply = response.content.strip()

        if not reply:
            return {"reply": FALLBACK_REPLY, "disclaimer_included": True}

        # Strict disclaimer detection — only true disclaimer phrases trigger bypass.
        # Avoid bypass via common words like "edukasi" that appear in normal medical text.
        disclaimer_patterns = [
            re.compile(r"bukan\s+(pengganti|pengganti\s+(saran|diagnosis|medis))", re.IGNORECASE),
            re.compile(r"(sifat|hanya)\s+(edukasi|edukatif)", re.IGNORECASE),
            re.compile(r"bukan\s+digantikan", re.IGNORECASE),
            re.compile(r"disclaimer", re.IGNORECASE),
            re.compile(r"konsultasikan\s+dengan\s+(bidan|dokter|tenaga\s+medis)", re.IGNORECASE),
            re.compile(r"bukan\s+(pengganti|pengganti)\s+(nasihat|saran)\s+(medis|bidan|dokter)", re.IGNORECASE),
        ]
        has_disclaimer = any(p.search(reply) for p in disclaimer_patterns)

        if not has_disclaimer:
            reply += DISCLAIMER

        logger.info(f"Chat reply generated ({len(reply)} chars)")
        return {"reply": reply, "disclaimer_included": True}

    except Exception as exc:
        logger.error(f"Chatbot LLM error: {type(exc).__name__}: {exc}")
        return {"reply": FALLBACK_REPLY, "disclaimer_included": True}
