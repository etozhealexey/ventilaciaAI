from typing import Any

from ventilacia_ai.models.schemas import CorrectionPayload
from ventilacia_ai.services.training_store import add_user_correction


def save_user_correction(payload: CorrectionPayload) -> dict[str, Any]:
    ok = add_user_correction(
        original_name=payload.original_name,
        corrected_code=payload.corrected_code,
        corrected_name=payload.corrected_name,
        corrected_unit=payload.corrected_unit,
    )
    if not ok:
        return {"success": False, "error": "Не удалось сохранить исправление"}
    return {"success": True}

