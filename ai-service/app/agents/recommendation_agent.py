"""
MaternIn AI Service — Recommendation Narrative LLM Agent (Triage Lapis 3)
==========================================================================
Menggunakan LangChain + GROQ API untuk menghasilkan narasi rekomendasi
yang mudah dipahami ibu hamil, berdasarkan risk_factors dan parameter klinis.

Guardrails (PRD Section 5 & 10):
  - LLM DILARANG mengoreksi/mengubah risk_badge dan aggregate_score
  - JANGAN kirim PII (nama, nomor HP) ke API LLM
  - Fallback teks jika LLM timeout/error
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings

logger = logging.getLogger("maternin.ai.agent.recommendation")

# ── Fallback text jika LLM gagal ─────────────────────────────────────────
FALLBACK_RECOMMENDATION = (
    "Detail rekomendasi saat ini tidak dapat ditampilkan. "
    "Silakan konsultasikan langsung dengan bidan Anda.\n\n"
    "⚕️ *Jawaban ini bersifat edukasi dan bukan pengganti saran atau diagnosis medis resmi.*"
)

# ── System prompt untuk narasi rekomendasi ────────────────────────────────
SYSTEM_PROMPT = """Kamu adalah asisten kesehatan ibu hamil bernama MaternIn.
Tugasmu HANYA menjelaskan hasil SKRINING AWAL kehamilan dalam bahasa Indonesia yang mudah dipahami ibu hamil dan keluarganya.

POSITIONING KAMU:
- Kamu adalah alat BANTU SKRINING untuk bidan, BUKAN alat diagnosis
- Keputusan klinis SELALU ada di tangan bidan/dokter
- Kamu membantu ibu memahami hasil skrining, BUKAN menggantikan bidan

ATURAN MUTLAK:
1. JANGAN PERNAH mengubah, mengoreksi, atau mempertanyakan nilai risk_badge atau aggregate_score yang diberikan.
2. JANGAN PERNAH membuat diagnosis medis sendiri.
3. JANGAN PERNAH menyebut nama pasien, nomor telepon, atau data pribadi lainnya.
4. JANGAN PERNAH mengarahkan pasien langsung ke IGD atau tindakan medis apapun.
   SELALU arahkan ke bidan: "Hubungi bidan Anda", "Bidan yang akan memutuskan".
5. Bingkai setiap pesan sebagai SKRINING AWAL — buka frasa dengan
   "Berdasarkan skrining awal MaternIn" atau "Hasil skrining menunjukkan".
6. Selalu akhiri dengan anjuran untuk konsultasi dengan bidan/dokter.

FORMAT JAWABAN:
- Untuk MERAH: Tekankan bahwa hasil ini BUKAN diagnosis — perlu VERIFIKASI BIDAN.
  Kalimat contoh: "Skrining menemukan indikasi yang perlu dievaluasi bidan sesegera mungkin.
  Bidan Anda yang akan menentukan langkah selanjutnya."
  JANGAN gunakan: "kondisi Anda adalah X", "perlu penanganan medis segera",
  "ke IGD", "jangan menunda".
- Untuk KUNING: Anjurkan pemeriksaan oleh bidan dalam waktu dekat. Nada perhatian.
- Untuk HIJAU: Berikan apresiasi, anjurkan tetap rutin ANC. Nada positif."""

USER_PROMPT_TEMPLATE = """Berdasarkan analisis kesehatan kehamilan, berikut hasilnya:

Status Risiko: {risk_badge_display}
Skor Risiko: {aggregate_score}/100
Usia Kehamilan: {gestational_age} minggu

Faktor Risiko yang Terdeteksi:
{risk_factors_text}

Buatkan narasi rekomendasi yang mudah dipahami ibu hamil (2-4 paragraf singkat).
Sertakan: penjelasan kondisi, saran tindakan spesifik, dan anjuran konsultasi bidan."""

# ── Badge display mapping ────────────────────────────────────────────────
BADGE_DISPLAY = {
    "merah": "🔴 RISIKO TINGGI",
    "kuning": "🟡 RISIKO SEDANG",
    "hijau": "🟢 RISIKO RENDAH",
}


def _get_llm():
    """Lazy init LangChain ChatGroq."""
    try:
        from langchain_groq import ChatGroq

        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=settings.groq_api_key,
            temperature=0.3,
            max_tokens=512,
            timeout=10,
            max_retries=2,
        )
    except ImportError:
        logger.warning("langchain-groq not installed, LLM agent disabled")
        return None
    except Exception as exc:
        logger.warning(f"Failed to init ChatGroq: {exc}")
        return None


async def generate_recommendation(
    risk_badge: str,
    aggregate_score: float,
    risk_factors: list[str],
    gestational_age_weeks: float | None = None,
    **kwargs: Any,
) -> str:
    """
    Generate narasi rekomendasi via LLM (Lapis 3).

    Args:
        risk_badge: "hijau" | "kuning" | "merah"
        aggregate_score: Skor risiko agregat (0-100)
        risk_factors: List faktor risiko klinis
        gestational_age_weeks: Usia kehamilan (minggu)

    Returns:
        Narasi rekomendasi dalam bahasa Indonesia.
    """
    llm = _get_llm()
    if llm is None:
        logger.warning("LLM not available, using fallback recommendation")
        return _generate_fallback(risk_badge, risk_factors, aggregate_score)

    # Build prompt — TANPA PII
    risk_factors_text = "\n".join(f"  • {f}" for f in risk_factors) if risk_factors else "  • Tidak ada faktor risiko signifikan"
    badge_display = BADGE_DISPLAY.get(risk_badge, risk_badge.upper())
    gestational_age = gestational_age_weeks or "tidak diketahui"

    user_message = USER_PROMPT_TEMPLATE.format(
        risk_badge_display=badge_display,
        aggregate_score=aggregate_score,
        risk_factors_text=risk_factors_text,
        gestational_age=gestational_age,
    )

    try:
        from langchain_core.messages import HumanMessage, SystemMessage

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ]

        response = await llm.ainvoke(messages)
        recommendation = response.content.strip()

        if not recommendation:
            return _generate_fallback(risk_badge, risk_factors, aggregate_score)

        logger.info(f"LLM recommendation generated ({len(recommendation)} chars)")
        return recommendation

    except Exception as exc:
        logger.error(f"LLM recommendation error: {type(exc).__name__}: {exc}")
        return _generate_fallback(risk_badge, risk_factors, aggregate_score)


def _generate_fallback(
    risk_badge: str,
    risk_factors: list[str],
    aggregate_score: float,
) -> str:
    """Fallback teks generik jika LLM tidak tersedia / error."""
    factors_text = ", ".join(risk_factors) if risk_factors else "Tidak ada faktor risiko signifikan"

    disclaimer = (
        "\n\n⚕️ *Jawaban ini bersifat edukasi dan bukan pengganti saran atau diagnosis medis resmi. "
        "Untuk penanganan lebih lanjut, konsultasikan dengan bidan atau dokter Anda.*"
    )

    if risk_badge == "merah":
        return (
            f"⚠️ Hasil skrining MaternIn menemukan indikasi yang perlu evaluasi bidan "
            f"sesegera mungkin (skor {aggregate_score:.0f}/100). "
            f"Faktor yang terdeteksi: {factors_text}. "
            f"Silakan hubungi bidan Anda untuk langkah selanjutnya. "
            f"Bidan Anda yang akan menentukan apakah perlu kunjungan IGD."
            f"{disclaimer}"
        )
    elif risk_badge == "kuning":
        return (
            f"⚡ Skrining menemukan beberapa hal yang perlu dipantau (skor {aggregate_score:.0f}/100). "
            f"Jadwalkan kunjungan ke bidan dalam waktu dekat untuk pemeriksaan lanjutan. "
            f"Faktor yang perlu dipantau: {factors_text}. "
            f"Istirahat cukup dan pantau gejala secara rutin."
            f"{disclaimer}"
        )
    else:
        return (
            f"✅ Hasil skrining menunjukkan kondisi baik (skor {aggregate_score:.0f}/100). "
            f"Lanjutkan pemeriksaan rutin sesuai jadwal ANC. "
            f"Jaga pola makan bergizi dan istirahat cukup. "
            f"Hubungi bidan jika muncul keluhan baru."
            f"{disclaimer}"
        )
