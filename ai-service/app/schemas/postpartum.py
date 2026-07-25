"""
Pydantic Schemas — Postpartum Evaluate Endpoint
=================================================
Kontrak request/response untuk POST /api/v1/postpartum/evaluate (P1).
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class BleedingLevel(str, Enum):
    normal = "normal"
    banyak = "banyak"
    sangat_banyak = "sangat_banyak"


class WoundCondition(str, Enum):
    baik = "baik"
    bau = "bau"
    bengkak_merah = "bengkak_merah"


class MoodFlag(str, Enum):
    baik = "baik"
    kadang_sedih = "kadang_sedih"
    sering_sedih = "sering_sedih"


class PostpartumLogEntry(BaseModel):
    """Sub-model: satu entri log harian postpartum."""
    day_number: int = Field(..., ge=1, le=42, description="Hari ke-N masa nifas (1-42)")
    bleeding_level: BleedingLevel = Field(..., description="Level perdarahan")
    fever: bool = Field(default=False, description="Apakah ada demam")
    wound_condition: WoundCondition = Field(default=WoundCondition.baik, description="Kondisi luka jahitan/C-section")
    headache_severe: bool = Field(default=False, description="Sakit kepala hebat")
    mood_flag: MoodFlag = Field(default=MoodFlag.baik, description="Indikator mood/mental")


class PostpartumEvaluateRequest(BaseModel):
    """Request body untuk POST /api/v1/postpartum/evaluate."""

    pregnancy_profile_id: str = Field(
        ..., description="UUID profil kehamilan pasien"
    )
    logs: list[PostpartumLogEntry] = Field(
        ...,
        min_length=1,
        description="Log harian postpartum (minimal 1 entri, bisa berupa histori beberapa hari)",
    )
    had_preeclampsia_history: bool = Field(
        default=False,
        description="Riwayat preeklampsia dari profil kehamilan",
    )
    bidan_phone: str | None = Field(
        default=None,
        description="Nomor WA bidan penanggung jawab (untuk alert darurat jika red flag)",
    )


class PostpartumEvaluateResponse(BaseModel):
    """Response body dari POST /api/v1/postpartum/evaluate."""

    red_flag_triggered: bool = Field(
        ..., description="Apakah ada red flag nifas yang terdeteksi"
    )
    reason: str = Field(
        default="",
        description="Alasan klinis pemicu red flag (kosong jika tidak ada)",
    )
    mental_health_flag: bool = Field(
        default=False,
        description="Indikasi potensi baby blues / gangguan kesehatan mental",
    )
