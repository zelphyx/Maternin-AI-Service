"""
MaternIn AI Service — NestJS Internal Webhook Client
======================================================
HTTP client untuk mengirim callback hasil kalkulasi ke endpoint internal NestJS:
  - POST /internal/risk-assessments
  - POST /internal/postpartum-flags

Spesifikasi (PRD Section 6):
  - Header wajib: X-Internal-Token + X-Request-Id
  - Timeout eksplisit: 5 detik
  - Retry otomatis: maks 3x dengan exponential backoff
  - Jika NestJS down setelah 3x retry: log error, JANGAN silent drop
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.config import settings
from app.schemas.nestjs_callback import PostpartumFlagCallback, RiskAssessmentCallback

logger = logging.getLogger("maternin.ai.service.nestjs_client")

# ── Konfigurasi Retry & Timeout ──────────────────────────────────────────
MAX_RETRIES = 3
TIMEOUT_SECONDS = 5.0
BACKOFF_BASE_SECONDS = 0.5  # 0.5s -> 1s -> 2s


async def _send_callback(
    endpoint_path: str,
    payload: dict[str, Any],
    request_id: str | None = None,
) -> dict[str, Any]:
    """
    Internal helper: kirim HTTP POST ke endpoint NestJS dengan retry + backoff.

    Args:
        endpoint_path: Path relatif (misal "/internal/risk-assessments").
        payload: JSON body yang dikirim.
        request_id: X-Request-Id untuk tracing lintas service.

    Returns:
        dict dengan "success" (bool), "status_code" (int|None), "detail" (str).
    """
    url = f"{settings.nestjs_internal_base_url}{endpoint_path}"

    headers = {
        "Content-Type": "application/json",
        "X-Internal-Token": settings.internal_service_token,
    }
    if request_id:
        headers["X-Request-Id"] = request_id

    last_error: Exception | None = None

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            async with httpx.AsyncClient(timeout=TIMEOUT_SECONDS) as client:
                response = await client.post(url, json=payload, headers=headers)

            if response.status_code in (200, 201):
                logger.info(
                    f"[{request_id}] Callback to {endpoint_path} succeeded "
                    f"(attempt {attempt}, status={response.status_code})"
                )
                return {
                    "success": True,
                    "status_code": response.status_code,
                    "detail": "Callback delivered successfully",
                }

            # Non-retryable HTTP errors (4xx client errors)
            if 400 <= response.status_code < 500:
                logger.error(
                    f"[{request_id}] Callback to {endpoint_path} rejected by NestJS "
                    f"(status={response.status_code}, body={response.text[:200]})"
                )
                return {
                    "success": False,
                    "status_code": response.status_code,
                    "detail": f"NestJS rejected callback: {response.status_code}",
                }

            # 5xx → retry
            logger.warning(
                f"[{request_id}] Callback to {endpoint_path} server error "
                f"(attempt {attempt}/{MAX_RETRIES}, status={response.status_code})"
            )
            last_error = Exception(f"HTTP {response.status_code}")

        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
            logger.warning(
                f"[{request_id}] Callback to {endpoint_path} connection failed "
                f"(attempt {attempt}/{MAX_RETRIES}): {type(exc).__name__}: {exc}"
            )
            last_error = exc

        # Exponential backoff sebelum retry berikutnya
        if attempt < MAX_RETRIES:
            import asyncio
            backoff = BACKOFF_BASE_SECONDS * (2 ** (attempt - 1))
            await asyncio.sleep(backoff)

    # Semua retry gagal — log error tapi JANGAN crash
    logger.error(
        f"[{request_id}] ❌ Callback to {endpoint_path} FAILED after {MAX_RETRIES} attempts. "
        f"Last error: {last_error}"
    )
    return {
        "success": False,
        "status_code": None,
        "detail": f"All {MAX_RETRIES} retry attempts failed: {last_error}",
    }


# ── Public API ───────────────────────────────────────────────────────────

async def post_risk_assessment_callback(
    payload: RiskAssessmentCallback,
    request_id: str | None = None,
) -> dict[str, Any]:
    """
    Kirim hasil analisis triage ke NestJS POST /internal/risk-assessments.

    Args:
        payload: RiskAssessmentCallback Pydantic model.
        request_id: X-Request-Id dari request awal.

    Returns:
        dict dengan status pengiriman.
    """
    logger.info(
        f"[{request_id}] Sending risk assessment callback: "
        f"profile={payload.pregnancy_profile_id}, badge={payload.risk_badge.value}"
    )
    return await _send_callback(
        endpoint_path="/internal/risk-assessments",
        payload=payload.model_dump(mode="json"),
        request_id=request_id,
    )


async def post_postpartum_flag_callback(
    payload: PostpartumFlagCallback,
    request_id: str | None = None,
) -> dict[str, Any]:
    """
    Kirim hasil evaluasi postpartum ke NestJS POST /internal/postpartum-flags.

    Args:
        payload: PostpartumFlagCallback Pydantic model.
        request_id: X-Request-Id dari request awal.

    Returns:
        dict dengan status pengiriman.
    """
    logger.info(
        f"[{request_id}] Sending postpartum flag callback: "
        f"profile={payload.pregnancy_profile_id}, red_flag={payload.red_flag_triggered}"
    )
    return await _send_callback(
        endpoint_path="/internal/postpartum-flags",
        payload=payload.model_dump(mode="json"),
        request_id=request_id,
    )
