"""
Pydantic Schemas — Triage Analyze Endpoint
============================================
Kontrak request/response untuk POST /api/v1/triage/analyze (P0).
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RiskBadge(str, Enum):
    """Enum status risiko — konsisten dengan skema NestJS."""
    hijau = "hijau"
    kuning = "kuning"
    merah = "merah"


class LatestAncData(BaseModel):
    """Sub-model: data ANC terakhir pasien."""
    systolic: int | None = Field(default=None, description="Tekanan sistolik (mmHg)")
    diastolic: int | None = Field(default=None, description="Tekanan diastolik (mmHg)")
    protein_urine: str | None = Field(default=None, description="Hasil protein urine (negatif/positif/positif_kuat)")
    weight_kg: float | None = Field(default=None, description="Berat badan (kg)")
    fundal_height_cm: float | None = Field(default=None, description="Tinggi fundus (cm)")
    platelet_count: float | None = Field(default=None, description="Jumlah trombosit")


class TriageAnalyzeRequest(BaseModel):
    """Request body untuk POST /api/v1/triage/analyze."""

    pregnancy_profile_id: str = Field(
        ..., description="UUID profil kehamilan pasien"
    )
    symptom_checkin_id: str = Field(
        ..., description="UUID checkin gejala yang sedang dianalisis"
    )
    answers: dict[str, Any] = Field(
        ...,
        description="Jawaban kuesioner adaptif (key: nama gejala, value: true/false/string severity)",
        examples=[{"bengkak_kaki": True, "sakit_kepala": "berat", "pandangan_kabur": False}],
    )
    conjunctiva_image_url: str | None = Field(
        default=None,
        description="URL gambar konjungtiva untuk deteksi anemia (opsional)",
    )
    latest_anc: LatestAncData | None = Field(
        default=None,
        description="Data ANC terakhir pasien",
    )
    has_preeclampsia_history: bool = Field(
        default=False,
        description="Riwayat preeklampsia pada kehamilan sebelumnya",
    )

    # Data tambahan untuk alert WA darurat
    bidan_phone: str | None = Field(
        default=None,
        description="Nomor WA bidan penanggung jawab (dikirim NestJS untuk alert darurat)",
    )


class TriageAnalyzeResponse(BaseModel):
    """Response body dari POST /api/v1/triage/analyze."""

    risk_badge: RiskBadge = Field(
        ..., description="Badge risiko akhir: hijau / kuning / merah"
    )
    aggregate_score: float = Field(
        ..., description="Skor risiko agregat (0-100)"
    )
    risk_factors: list[str] = Field(
        default_factory=list,
        description="Faktor risiko klinis yang terdeteksi (explainable, dapat ditelusuri)",
    )
    recommendation_text: str = Field(
        default="",
        description="Narasi rekomendasi yang dihasilkan LLM (Lapis 3)",
    )
    triage_score: float | None = Field(
        default=None,
        description="Skor rule-based triage engine (Lapis 1)",
    )
    anemia_probability: float | None = Field(
        default=None,
        description="Probabilitas anemia dari CV model (0.0-1.0)",
    )
    preeclampsia_probability: float | None = Field(
        default=None,
        description="Probabilitas preeklampsia dari LR model (0.0-1.0)",
    )
    alert_delivery_status: str | None = Field(
        default=None,
        description="Status pengiriman WA skrining: sent | failed | not_triggered | pending_bidan_review",
    )
    anemia_is_mock: bool = Field(
        default=False,
        description="True jika anemia_probability dari mock placeholder, bukan model nyata. "
        "Ini penting untuk transparansi — nilai 0.25 BUKAN hasil inferensi.",
    )
    bidan_review_required: bool = Field(
        default=False,
        description="True jika risk_badge==merah dan menunggu konfirmasi bidan. "
        "WA alert TIDAK dikirim otomatis — bidan harus verify dulu via endpoint "
        "/triage/{id}/bidan-confirm.",
    )
    disclaimer: str = Field(
        default="Hasil ini adalah SKRINING OTOMATIS, BUKAN diagnosis medis. "
                "Keputusan klinis akhir ada di tangan bidan/dokter. "
                "Selalu verifikasi hasil ini dengan bidan penanggung jawab.",
        description="Disclaimer wajib — bidan-assist tool, bukan diagnostic tool. "
        "Machine-readable contract untuk consumer.",
    )
    screening_not_diagnosis: bool = Field(
        default=True,
        description="Selalu True untuk rilis bidan-assist. Reserve untuk mode diagnostic di masa depan.",
    )
