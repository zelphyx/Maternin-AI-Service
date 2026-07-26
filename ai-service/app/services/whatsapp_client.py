"""
MaternIn AI Service — Emergency WhatsApp Alert Client (Fonnte)
===============================================================
Client untuk mengirim peringatan darurat via WhatsApp ke bidan
penanggung jawab saat risk_badge == "merah".

Spesifikasi (PRD Section 7):
  - Trigger: risk_badge == "merah" dari /api/v1/triage/analyze
  - API: Fonnte (https://api.fonnte.com/send)
  - Retry: maks 3x jika API Fonnte gagal
  - Jika tetap gagal: tandai alert_delivery_status = "failed", JANGAN crash
  - Data nomor bidan dikirim NestJS di request payload (bidan_phone)
"""

from __future__ import annotations

import asyncio
import logging

import httpx

from app.core.config import settings

logger = logging.getLogger("maternin.ai.service.whatsapp")

# ── Konfigurasi ──────────────────────────────────────────────────────────
FONNTE_SEND_URL = "https://api.fonnte.com/send"
MAX_RETRIES = 3
TIMEOUT_SECONDS = 10.0
BACKOFF_BASE_SECONDS = 1.0  # 1s -> 2s -> 4s


async def send_emergency_alert(
    phone_number: str,
    message: str,
    request_id: str | None = None,
) -> str:
    """
    Kirim peringatan darurat via WhatsApp (Fonnte) ke bidan.

    Args:
        phone_number: Nomor WA bidan penanggung jawab (format: 62xxx).
        message: Isi pesan darurat.
        request_id: X-Request-Id untuk tracing.

    Returns:
        Status pengiriman: "sent" | "failed" | "no_phone"
    """
    if not phone_number:
        logger.warning(
            f"[{request_id}] ⚠️ Emergency WA alert skipped: no bidan_phone provided. "
            f"NestJS harus mengirim bidan_phone di request payload."
        )
        return "no_phone"

    # Sanitize nomor telepon
    clean_phone = phone_number.strip().replace("+", "").replace("-", "").replace(" ", "")
    if clean_phone.startswith("0"):
        clean_phone = "62" + clean_phone[1:]

    headers = {
        "Authorization": settings.fonnte_api_key,
    }

    payload = {
        "target": clean_phone,
        "message": message,
    }
    # Only add countryCode if the number doesn't already start with "62"
    if not clean_phone.startswith("62"):
        payload["countryCode"] = "62"

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                response = await client.post(
                    FONNTE_SEND_URL,
                    data=payload,
                    headers=headers,
                )

            # Fonnte mengembalikan JSON {"status": true/false, ...}
            if response.status_code == 200:
                resp_data = response.json()
                if resp_data.get("status"):
                    logger.info(
                        f"[{request_id}] ✅ Emergency WA alert SENT to {clean_phone} "
                        f"(attempt {attempt})"
                    )
                    return "sent"
                else:
                    # Fonnte menolak (misal: saldo habis, nomor invalid)
                    detail = resp_data.get("reason", resp_data.get("detail", "unknown"))
                    logger.warning(
                        f"[{request_id}] Fonnte rejected message "
                        f"(attempt {attempt}/{MAX_RETRIES}): {detail}"
                    )
                    last_error = Exception(f"Fonnte rejected: {detail}")
            else:
                logger.warning(
                    f"[{request_id}] Fonnte HTTP error "
                    f"(attempt {attempt}/{MAX_RETRIES}): status={response.status_code}"
                )
                last_error = Exception(f"HTTP {response.status_code}")

        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            logger.warning(
                f"[{request_id}] Fonnte connection failed "
                f"(attempt {attempt}/{MAX_RETRIES}): {type(exc).__name__}: {exc}"
            )
            last_error = exc

        # Exponential backoff sebelum retry berikutnya
        if attempt < MAX_RETRIES:
            backoff = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            await asyncio.sleep(backoff)

    # Semua retry gagal — log tapi JANGAN crash
    logger.error(
        f"[{request_id}] ❌ Emergency WA alert to {clean_phone} FAILED "
        f"after {MAX_RETRIES} attempts. Last error: {last_error}"
    )
    return "failed"


def build_emergency_message(
    risk_factors: list[str],
    aggregate_score: float,
    pregnancy_profile_id: str,
) -> str:
    """
    Buat template pesan skrining WA untuk bidan.

    IMPORTANT: Pesan ini adalah SKRINING, BUKAN diagnosis.
    Bidan harus verifikasi langsung sebelum mengambil tindakan klinis.
    """
    factors_text = "\n".join(f"  • {f}" for f in risk_factors) if risk_factors else "  • (Tidak ada detail)"

    return (
        f"🔔 *Skrining MaternIn — Perlu Verifikasi Bidan*\n"
        f"━━━━━━━━━━━━━━━━━\n"
        f"Sistem skrining AI mendeteksi *indikasi risiko tinggi* pada seorang pasien.\n"
        f"*Mohon verifikasi langsung* sebelum mengambil tindakan klinis.\n"
        f"\n"
        f"Indikator skrining (skor: *{aggregate_score:.0f}/100*):\n"
        f"{factors_text}\n"
        f"\n"
        f"⚠️ *Ini adalah hasil skrining, BUKAN diagnosis medis.*\n"
        f"Keputusan klinis akhir ada di tangan Bidan.\n"
        f"\n"
        f"Buka detail kasus di aplikasi MaternIn:\n"
        f"https://maternin.app/bidan/cases/{pregnancy_profile_id}\n"
        f"━━━━━━━━━━━━━━━━━\n"
        f"_MaternIn AI · auto-generated screening alert_\n"
        f"_Pesan ini hanya untuk tenaga kesehatan profesional._"
    )
