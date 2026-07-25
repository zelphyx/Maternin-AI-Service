"""
MaternIn AI Service — Configuration & Environment Validation
=============================================================
Menggunakan pydantic-settings untuk memuat dan memvalidasi .env.
Aplikasi akan fail-fast (gagal start) jika variabel wajib kosong.
"""

from pydantic_settings import BaseSettings
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """Konfigurasi utama AI Service — semua variabel sensitif wajib dari .env."""

    # --- API Keys ---
    groq_api_key: str = Field(
        ...,
        min_length=1,
        description="API key untuk GROQ LLM (wajib)",
    )
    fonnte_api_key: str = Field(
        ...,
        min_length=1,
        description="API key untuk Fonnte WhatsApp (wajib, khusus alert darurat)",
    )

    # --- Inter-Service Auth ---
    internal_service_token: str = Field(
        ...,
        min_length=32,
        description="Shared secret antara AI Service & NestJS (min 32 karakter)",
    )

    # --- NestJS Backend URL ---
    nestjs_internal_base_url: str = Field(
        default="http://localhost:3000",
        description="Base URL endpoint internal NestJS (tanpa trailing slash)",
    )

    # --- Model Artifacts ---
    model_artifact_dir: str = Field(
        default="./app/model_artifacts",
        description="Path ke folder artefak model terkompresi",
    )

    # --- Logging ---
    log_level: str = Field(
        default="info",
        description="Level logging (debug, info, warning, error, critical)",
    )

    @field_validator("nestjs_internal_base_url")
    @classmethod
    def strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        allowed = {"debug", "info", "warning", "error", "critical"}
        if v.lower() not in allowed:
            raise ValueError(f"log_level must be one of {allowed}, got '{v}'")
        return v.lower()

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


# Singleton — diimpor di seluruh aplikasi
settings = Settings()
