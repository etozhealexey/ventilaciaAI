from typing import Optional, List

from pydantic import BaseModel


class MatchItem(BaseModel):
    """Элемент заявки (позиция, количество, единица измерения)."""

    name: str
    quantity: Optional[str] = ""
    unit: Optional[str] = ""


class MatchRequest(BaseModel):
    """Запрос на сопоставление списка позиций."""

    items: List[MatchItem]


class CorrectionPayload(BaseModel):
    """Тело запроса для сохранения исправления пользователем."""

    original_name: str
    corrected_code: str
    corrected_name: Optional[str] = None
    corrected_unit: Optional[str] = None


