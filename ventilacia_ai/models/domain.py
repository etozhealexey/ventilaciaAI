from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class MatchResult:
    """Доменная модель результата сопоставления одной позиции."""

    original_name: str
    quantity: str
    matched_code: Optional[str]
    matched_name: Optional[str]
    matched_unit: Optional[str]
    confidence: float
    reason: str


@dataclass
class Correction:
    """Исправление пользователя (corrections)."""

    original_name: str
    corrected_code: str
    corrected_name: Optional[str]
    corrected_unit: Optional[str]
    timestamp: Optional[datetime] = None


