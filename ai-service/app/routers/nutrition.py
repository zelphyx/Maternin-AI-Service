"""
MaternIn AI Service — Nutrition Parse Router
===============================================
POST /api/v1/nutrition/parse (P2)
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from app.core.auth import get_request_id, verify_internal_token
from app.agents.nutrition_parser import parse_nutrition
from app.schemas.nutrition import (
    NutritionParseRequest,
    NutritionParseResponse,
    ParsedFoodItem,
)

logger = logging.getLogger("maternin.ai.router.nutrition")

router = APIRouter(
    prefix="/api/v1/nutrition",
    tags=["Nutrition"],
    dependencies=[Depends(verify_internal_token)],
)


@router.post(
    "/parse",
    response_model=NutritionParseResponse,
    summary="Parse laporan makan harian",
    description=(
        "Mengekstrak bahan makanan dan estimasi porsi dari teks bebas "
        "(misal dari pesan WhatsApp). Estimasi bersifat perkiraan kasar."
    ),
)
async def nutrition_parse(
    request: NutritionParseRequest,
    request_id: str | None = Depends(get_request_id),
) -> NutritionParseResponse:
    """Parse teks makanan dan kembalikan structured items."""
    logger.info(
        f"[{request_id}] Nutrition parse: "
        f"profile={request.pregnancy_profile_id}, "
        f"message_len={len(request.raw_message)}"
    )

    result = await parse_nutrition(
        raw_message=request.raw_message,
        pregnancy_profile_id=request.pregnancy_profile_id,
    )

    parsed_items = [
        ParsedFoodItem(name=item["name"], portion_estimate=item["portion_estimate"])
        for item in result.get("parsed_items", [])
    ]

    logger.info(
        f"[{request_id}] Nutrition result: "
        f"items={len(parsed_items)}"
    )

    return NutritionParseResponse(
        parsed_items=parsed_items,
        insight_text=result.get("insight_text", ""),
    )
