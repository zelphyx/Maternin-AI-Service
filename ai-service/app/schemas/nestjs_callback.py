"""
Pydantic Schemas — NestJS Internal Callback Payloads
======================================================
Payload yang dikirim AI Service ke endpoint internal NestJS:
  - POST /internal/risk-assessments
  - POST /internal/postpartum-flags
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.triage import RiskBadge


class RiskAssessmentCallback(BaseModel):
    """Payload callback ke NestJS POST /internal/risk-assessments."""

    pregnancy_profile_id: str = Field(
        ..., description="UUID profil kehamilan pasien"
    )
    symptom_checkin_id: str | None = Field(
        default=None, description="UUID checkin gejala terkait"
    )
    triage_score: float = Field(
        ..., description="Skor dari triage engine (Lapis 1)"
    )
    anemia_probability: float | None = Field(
        default=None, description="Probabilitas anemia dari CV model (0.0-1.0)"
    )
    preeclampsia_probability: float | None = Field(
        default=None, description="Probabilitas preeklampsia dari LR model (0.0-1.0)"
    )
    aggregate_score: float = Field(
        ..., description="Skor risiko agregat akhir (0-100)"
    )
    risk_badge: RiskBadge = Field(
        ..., description="Badge risiko akhir: hijau / kuning / merah"
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Faktor risiko klinis yang terdeteksi",
    )
    recommendation_text: str = Field(
        default="", description="Narasi rekomendasi dari LLM"
    )
    alert_delivery_status: str | None = Field(
        default=None,
        description="Status pengiriman WA darurat: sent / failed / not_triggered",
    )
    anemia_is_mock: bool = Field(
        default=False,
        description="True jika anemia_probability dari mock placeholder. NestJS bisa "
        "menandai hasil anemia sebagai 'unverified' di database.",
    )


class PostpartumFlagCallback(BaseModel):
    """Payload callback ke NestJS POST /internal/postpartum-flags."""

    pregnancy_profile_id: str = Field(
        ..., description="UUID profil kehamilan pasien"
    )
    red_flag_triggered: bool = Field(
        ..., description="Apakah red flag nifas terdeteksi"
    )
    reason: str = Field(
        default="", description="Alasan klinis pemicu red flag"
    )
    mental_health_flag: bool = Field(
        default=False, description="Indikasi baby blues / gangguan mental"
    )
