"""
Pydantic Schemas — Nutrition Parse Endpoint
=============================================
Kontrak request/response untuk POST /api/v1/nutrition/parse (P2).
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ParsedFoodItem(BaseModel):
    """Sub-model: satu item makanan hasil parsing."""
    name: str = Field(..., description="Nama bahan makanan")
    portion_estimate: str = Field(..., description="Estimasi porsi (misal: '1 centong')")


class NutritionParseRequest(BaseModel):
    """Request body untuk POST /api/v1/nutrition/parse."""

    pregnancy_profile_id: str = Field(
        ..., description="UUID profil kehamilan pasien"
    )
    raw_message: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Teks bebas laporan makan dari WhatsApp",
    )


class NutritionParseResponse(BaseModel):
    """Response body dari POST /api/v1/nutrition/parse."""

    parsed_items: list[ParsedFoodItem] = Field(
        default_factory=list,
        description="Daftar item makanan hasil parsing",
    )
    insight_text: str = Field(
        default="",
        description="Ringkasan insight gizi (estimasi, bukan angka presisi)",
    )
