"""
MaternIn AI Service — Custom LLM Client (httpx-based)
=====================================================
Bypasses langchain to avoid Cloudflare blocking issues.
Works with any OpenAI-compatible API (labs.inxorastudio.com, Groq, etc.)
Uses httpx directly with full header control.
"""

from __future__ import annotations

import logging
from typing import Any

from app.core.config import settings

logger = logging.getLogger("maternin.ai.llm_client")

FALLBACK_RESPONSES = {
    "recommendation_hijau": "✅ Hasil skrining menunjukkan kondisi baik. Lanjutkan pemeriksaan rutin sesuai jadwal ANC. Jaga pola makan bergizi dan istirahat cukup. Hubungi bidan jika muncul keluhan baru.\n\n⚕️ *Jawaban ini bersifat edukasi dan bukan pengganti saran atau diagnosis medis resmi.*",
    "recommendation_kuning": "⚡ Skrining menemukan beberapa hal yang perlu dipantau. Jadwalkan kunjungan ke bidan dalam waktu dekat untuk pemeriksaan lanjutan.\n\n⚕️ *Jawaban ini bersifat edukasi dan bukan pengganti saran atau diagnosis medis resmi.*",
    "recommendation_merah": "⚠️ Hasil skrining menemukan indikasi yang perlu evaluasi bidan sesegera mungkin. Silakan hubungi bidan Anda untuk langkah selanjutnya.\n\n⚕️ *Jawaban ini bersifat edukasi dan bukan pengganti saran atau diagnosis medis resmi.*",
}


def _make_llm_request(messages: list[dict[str, str]], **kwargs) -> str | None:
    """
    Make an LLM request via httpx to the configured AI API.
    Returns the response content string, or None on failure.
    """
    if not settings.ai_api_key:
        logger.warning("AI_API_KEY not configured, skipping LLM request")
        return None

    try:
        import httpx

        url = f"{settings.ai_api_base_url.rstrip('/')}/chat/completions"
        model = settings.ai_model
        timeout = kwargs.get("timeout", settings.ai_timeout_seconds)
        max_tokens = kwargs.get("max_tokens", 512)
        temperature = kwargs.get("temperature", 0.3)

        # httpx streaming or regular — use sync client for simplicity
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(
                url,
                json={
                    "model": model,
                    "messages": messages,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
                headers={
                    "Authorization": f"Bearer {settings.ai_api_key}",
                    "Content-Type": "application/json",
                },
            )
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    except Exception as exc:
        logger.error(f"LLM request failed: {type(exc).__name__}: {exc}")
        return None


# ── Recommendation ──────────────────────────────────────────────────────
async def generate_recommendation(
    risk_badge: str,
    aggregate_score: float,
    risk_factors: list[str],
    gestational_age_weeks: float | None = None,
    **kwargs: Any,
) -> str:
    """Generate narasi rekomendasi via LLM."""
    risk_factors_text = "\n".join(f"  • {f}" for f in risk_factors) if risk_factors else "  • Tidak ada faktor risiko signifikan"
    gestational_age = gestational_age_weeks or "tidak diketahui"

    badge_display = {
        "merah": "🔴 RISIKO TINGGI",
        "kuning": "🟡 RISIKO SEDANG",
        "hijau": "🟢 RISIKO RENDAH",
    }.get(risk_badge, risk_badge.upper())

    system = """Kamu adalah asisten kesehatan ibu hamil bernama MaternIn.
Tugasmu HANYA menjelaskan hasil SKRINING AWAL kehamilan dalam bahasa Indonesia yang mudah dipahami.

ATURAN MUTLAK:
1. JANGAN PERNAH mengubah risk_badge atau aggregate_score yang diberikan.
2. JANGAN PERNAH membuat diagnosis medis sendiri.
3. JANGAN PERNAH menyebut nama pasien atau data pribadi lainnya.
4. JANGAN PERNAH mengarahkan langsung ke IGD — SELALU arahkan ke bidan.
5. Selalu akhiri dengan anjuran konsultasi bidan/dokter.
6. Untuk MERAH: tekankan BUKAN diagnosis, perlu verifikasi bidan.
7. Untuk KUNING: anjurkan pemeriksaan dekat.
8. Untuk HIJAU: berikan apresiasi, anjurkan ANC rutin."""

    user = f"""Hasil skrining kehamilan:

Status Risiko: {badge_display}
Skor Risiko: {aggregate_score}/100
Usia Kehamilan: {gestational_age} minggu
Faktor Risiko:
{risk_factors_text}

Buatkan narasi 2-4 paragraf yang mudah dipahami ibu hamil."""

    response = _make_llm_request(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=512,
        temperature=0.3,
    )

    if response:
        logger.info(f"LLM recommendation generated ({len(response)} chars)")
        return response

    # Fallback
    key = f"recommendation_{risk_badge}"
    return FALLBACK_RESPONSES.get(key, FALLBACK_RESPONSES["recommendation_hijau"])


# ── Chatbot ─────────────────────────────────────────────────────────────
async def chat_reply(
    message: str,
    pregnancy_profile_id: str,
    grounding_context: str = "",
    **kwargs: Any,
) -> dict[str, Any]:
    """Chat reply via LLM."""
    logger.info(f"Chat request: profile={pregnancy_profile_id}, len={len(message)}")

    system = f"""Kamu adalah MaternIn, asisten edukasi kesehatan ibu hamil di Indonesia.
Jawab dalam bahasa Indonesia yang sederhana dan hangat.

ATURAN:
1. SELALU akhiri dengan disclaimer: "Jawaban ini bersifat edukasi dan bukan pengganti saran atau diagnosis medis resmi."
2. JANGAN buat diagnosis medis.
3. JANGAN berikan angka skor risiko sendiri.
4. JANGAN resepkan obat.
5. Jika gejala darurat (perdarahan, kejang, pandangan kabur), anjurkan ke IGD terdekat.

KONTEKS GROUNDING:
{grounding_context or 'Tidak ada konteks tambahan.'}"""

    response = _make_llm_request(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": message},
        ],
        max_tokens=768,
        temperature=0.4,
    )

    disclaimer = "\n\n⚕️ *Jawaban ini bersifat edukasi dan bukan pengganti saran atau diagnosis medis resmi.*"

    if response:
        if "disclaimer" not in response.lower() and "bukan pengganti" not in response.lower():
            response += disclaimer
        logger.info(f"Chat reply generated ({len(response)} chars)")
        return {"reply": response, "disclaimer_included": True}

    return {
        "reply": "Maaf, saat ini saya tidak dapat memproses pertanyaan Anda. Silakan coba lagi nanti atau konsultasikan langsung dengan bidan Anda." + disclaimer,
        "disclaimer_included": True,
    }


# ── Nutrition Parser ──────────────────────────────────────────────────────
async def parse_nutrition(
    raw_message: str,
    pregnancy_profile_id: str,
) -> dict[str, Any]:
    """Parse nutrition via LLM structured output."""
    logger.info(f"Nutrition parse: profile={pregnancy_profile_id}, len={len(raw_message)}")

    system = """Kamu parser makanan Indonesia untuk ibu hamil.
Output HARUS dalam JSON valid:
{
  "items": [{"name": "nama", "portion_estimate": "porsi"}],
  "insight": "ringkasan gizi untuk ibu hamil (sebut ini estimasi)"
}"""

    response = _make_llm_request(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Parse: {raw_message}"},
        ],
        max_tokens=512,
        temperature=0.1,
    )

    if not response:
        return _parse_fallback(raw_message)

    try:
        import json
        # Extract JSON
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            data = json.loads(response[start:end])
            items = [{"name": i["name"], "portion_estimate": i.get("portion_estimate", "1 porsi")} for i in data.get("items", [])]
            insight = data.get("insight", "")
            if insight and "estimasi" not in insight.lower():
                insight += " (Catatan: estimasi kasar, bukan pengukuran presisi.)"
            return {"parsed_items": items, "insight_text": insight}
    except Exception:
        pass

    return _parse_fallback(raw_message)


def _parse_fallback(raw_message: str) -> dict[str, Any]:
    common = {"nasi": "1 centong", "telur": "1 butir", "sayur": "1 mangkuk", "bayam": "1 mangkuk",
              "tempe": "2 potong", "tahu": "2 potong", "ayam": "1 potong", "ikan": "1 potong",
              "susu": "1 gelas", "pisang": "1 buah", "roti": "1 lembar", "bubur": "1 mangkuk"}
    items = [{"name": f, "portion_estimate": p} for f, p in common.items() if f in raw_message.lower()]
    return {
        "parsed_items": items,
        "insight_text": f"Terdeteksi {len(items)} jenis makanan. Estimasi di atas perkiraan kasar. Konsultasikan dengan bidan atau ahli gizi.",
    }


# ── Visit Brief ─────────────────────────────────────────────────────────
async def generate_visit_brief(
    anc_history: list[dict[str, Any]],
    risk_assessments: list[dict[str, Any]],
    postpartum_logs: list[dict[str, Any]],
    pregnancy_profile_id: str,
) -> str:
    """Generate visit brief via LLM."""
    import json
    logger.info(f"Visit brief: profile={pregnancy_profile_id}, anc={len(anc_history)}")

    system = """Kamu sistem ringkasan medis untuk bidan Indonesia.
Buat ringkasan 2-3 kalimat dari data pasien. Fokus pada kondisi terkini, red flags, dan tren risiko.
JANGAN tambahkan interpretasi yang tidak ada di data. Bahasa Indonesia formal."""

    anc = json.dumps(anc_history[-5:], ensure_ascii=False, default=str) if anc_history else "Tidak ada"
    risk = json.dumps(risk_assessments[-5:], ensure_ascii=False, default=str) if risk_assessments else "Tidak ada"
    pp = json.dumps(postpartum_logs[-5:], ensure_ascii=False, default=str) if postpartum_logs else "Tidak ada"

    user = f"""Data pasien:\nANC: {anc}\nRisk: {risk}\nPostpartum: {pp}\n\nRingkasan 2-3 kalimat untuk bidan."""

    response = _make_llm_request(
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        max_tokens=256,
        temperature=0.2,
    )

    if response:
        logger.info(f"Visit brief generated ({len(response)} chars)")
        return response

    # Fallback
    parts = []
    if risk_assessments:
        latest = risk_assessments[-1]
        parts.append(f"Risk terakhir: {latest.get('risk_badge', '?')} (skor {latest.get('aggregate_score', '?')}/100).")
    if anc_history:
        parts.append(f"{len(anc_history)} kunjungan ANC tercatat.")
    if postpartum_logs:
        parts.append(f"{len(postpartum_logs)} log postpartum.")
    return " ".join(parts) if parts else "Tidak ada data tersedia."
