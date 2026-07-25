"""
MaternIn AI Service — Auth & Tracing Dependencies
===================================================
1. verify_internal_token: FastAPI Dependency untuk validasi header X-Internal-Token.
2. get_request_id: Ekstraksi X-Request-Id dari header masuk untuk tracing lintas service.
"""

from fastapi import Depends, Header, HTTPException, Request, status

from app.core.config import settings


async def verify_internal_token(
    x_internal_token: str = Header(
        ...,
        alias="X-Internal-Token",
        description="Shared secret dari NestJS — wajib ada di setiap request",
    ),
) -> str:
    """
    Dependency: validasi header X-Internal-Token.
    Mengembalikan token jika valid; raise 401 jika salah/kosong.
    """
    if not x_internal_token or x_internal_token != settings.internal_service_token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing X-Internal-Token",
        )
    return x_internal_token


async def get_request_id(
    x_request_id: str | None = Header(
        default=None,
        alias="X-Request-Id",
        description="ID request dari NestJS untuk tracing lintas service",
    ),
) -> str | None:
    """
    Dependency: ekstraksi X-Request-Id dari header masuk.
    Bisa None jika NestJS tidak mengirimkan (fallback logging).
    """
    return x_request_id
