"""
MaternIn AI Service — Nutrition NLP Parser Agent
==================================================
Mengekstrak bahan makanan dan estimasi porsi dari teks bebas (WhatsApp).
Menggunakan LLM structured JSON output + TKPI database fallback.

Guardrails (PRD Section 4.6):
  - insight_text wajib menyebut bahwa ini estimasi, bukan angka presisi
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from app.core.config import settings

logger = logging.getLogger("maternin.ai.agent.nutrition")

# ── TKPI lookup (fallback jika LLM down) ─────────────────────────────────
_tkpi_data: list[dict] | None = None


def _tkpi_path() -> str:
    """
    Resolve path ke TKPI nutrition CSV.
    Production (HF Space): MATERIN_DATA_DIR/nutrition/<file>
    Dev (local): ../../../datasets/tkpi_nutrition/<file>
    """
    base = os.environ.get("MATERIN_DATA_DIR")
    filename = "tkpi_indonesian_food_master_300.csv"
    if base:
        return os.path.join(base, "nutrition", filename)
    return os.path.join(
        os.path.dirname(__file__), "..", "..", "..", "datasets",
        "tkpi_nutrition", filename
    )


def _load_tkpi() -> list[dict]:
    """Load TKPI Indonesian food database."""
    global _tkpi_data
    if _tkpi_data is not None:
        return _tkpi_data

    tkpi_path = _tkpi_path()

    _tkpi_data = []
    try:
        if os.path.exists(tkpi_path):
            import csv
            with open(tkpi_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                _tkpi_data = list(reader)
            logger.info(f"Loaded {len(_tkpi_data)} TKPI food items")
    except Exception as exc:
        logger.warning(f"Failed to load TKPI: {exc}")

    return _tkpi_data


def _get_llm():
    """Lazy init ChatGroq."""
    try:
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.groq_api_key,
            temperature=0.1,
            max_tokens=512,
            timeout=10,
            max_retries=2,
        )
    except Exception:
        return None


SYSTEM_PROMPT = """Kamu adalah parser makanan Indonesia untuk ibu hamil.

TUGAS: Dari pesan teks bebas, ekstrak daftar makanan dan estimasi porsi.

ATURAN:
1. Output HARUS dalam format JSON valid.
2. Estimasi porsi menggunakan satuan umum Indonesia (centong, butir, mangkuk, potong, gelas, dll).
3. Jika porsi tidak disebutkan, estimasi 1 porsi standar.
4. Fokus pada makanan yang disebutkan, jangan menambahkan makanan yang tidak ada.

FORMAT OUTPUT (JSON saja, tanpa penjelasan lain):
{
  "items": [
    {"name": "nama makanan", "portion_estimate": "estimasi porsi"}
  ],
  "insight": "ringkasan singkat tentang kecukupan gizi untuk ibu hamil (1-2 kalimat, SEBUTKAN bahwa ini estimasi kasar)"
}"""


async def parse_nutrition(
    raw_message: str,
    pregnancy_profile_id: str,
) -> dict[str, Any]:
    """
    Parse teks makanan → structured food items + insight.

    Returns:
        dict with "parsed_items" (list) and "insight_text" (str)
    """
    logger.info(
        f"Nutrition parse: profile={pregnancy_profile_id}, "
        f"message_len={len(raw_message)}"
    )

    llm = _get_llm()
    if llm is None:
        return _parse_fallback(raw_message)

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        response = await llm.ainvoke([
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=f"Pesan makanan: \"{raw_message}\""),
        ])

        result = _extract_json(response.content)
        if result:
            items = [
                {"name": item["name"], "portion_estimate": item.get("portion_estimate", "1 porsi")}
                for item in result.get("items", [])
            ]
            insight = result.get("insight", "")

            # Guardrail: pastikan insight menyebut estimasi
            if insight and "estimasi" not in insight.lower():
                insight += " (Catatan: nilai porsi di atas merupakan estimasi kasar, bukan hasil pengukuran presisi.)"

            logger.info(f"Nutrition parsed: {len(items)} items")
            return {"parsed_items": items, "insight_text": insight}

    except Exception as exc:
        logger.error(f"Nutrition LLM error: {type(exc).__name__}: {exc}")

    return _parse_fallback(raw_message)


def _extract_json(text: str) -> dict | None:
    """Extract JSON dari response LLM (bisa ada teks sebelum/sesudah JSON)."""
    try:
        # Cari JSON block
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except json.JSONDecodeError:
        pass
    return None


def _parse_fallback(raw_message: str) -> dict[str, Any]:
    """Fallback: simple keyword-based parsing tanpa LLM."""
    common_foods = {
        "nasi": "1 centong",
        "telur": "1 butir",
        "sayur": "1 mangkuk kecil",
        "bayam": "1 mangkuk kecil",
        "tempe": "2 potong",
        "tahu": "2 potong",
        "ayam": "1 potong",
        "ikan": "1 potong",
        "susu": "1 gelas",
        "buah": "1 porsi",
        "pisang": "1 buah",
        "roti": "1 lembar",
        "mie": "1 bungkus",
        "soto": "1 mangkuk",
        "bubur": "1 mangkuk",
        "kangkung": "1 mangkuk kecil",
        "wortel": "1 mangkuk kecil",
        "daging": "1 potong",
        "jeruk": "1 buah",
        "apel": "1 buah",
    }

    msg_lower = raw_message.lower()
    items = []
    for food, portion in common_foods.items():
        if food in msg_lower:
            items.append({"name": food, "portion_estimate": portion})

    insight = (
        f"Terdeteksi {len(items)} jenis makanan dari pesan Anda. "
        "Estimasi porsi di atas merupakan perkiraan kasar berdasarkan porsi standar Indonesia, "
        "bukan hasil pengukuran presisi. Untuk kebutuhan gizi spesifik ibu hamil, "
        "konsultasikan dengan bidan atau ahli gizi."
    )

    return {"parsed_items": items, "insight_text": insight}
