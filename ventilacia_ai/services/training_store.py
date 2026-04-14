import json
from datetime import datetime
from typing import Any

from ventilacia_ai.services.config_service import TRAINING_DATA_FILE
from ventilacia_ai.services.text_utils import normalize_name

_MAX_CORRECTIONS = 5_000


def load_training_data() -> dict[str, Any]:
    """Загружает данные обучения (corrections) из файла."""
    try:
        with open(TRAINING_DATA_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        data.setdefault("corrections", [])
        return data
    except FileNotFoundError:
        return {"corrections": []}
    except Exception as e:
        print(f"Ошибка загрузки истории обучения: {e}")
        return {"corrections": []}


def save_training_data(training_data: dict[str, Any]) -> bool:
    """Сохраняет данные обучения в файл."""
    try:
        with open(TRAINING_DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(training_data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"Ошибка сохранения истории обучения: {e}")
        return False


def get_training_examples_for_prompt(
    item_name: str | None = None, limit: int = 100
) -> list[dict[str, Any]]:
    """
    Возвращает примеры из corrections для few-shot промпта.
    При наличии item_name — ранжирует по embedding-близости.
    """
    training_data = load_training_data()
    corrections = training_data.get("corrections", [])
    if not corrections:
        return []

    correction_examples: list[dict[str, Any]] = []
    for correction in corrections:
        correction_examples.append(
            {
                "original_name": correction.get("original_name", ""),
                "matched_code": correction.get("corrected_code"),
                "matched_name": correction.get("corrected_name", ""),
                "matched_unit": correction.get("corrected_unit", ""),
                "confidence": 1.0,
                "is_corrected": True,
                "timestamp": correction.get("timestamp", ""),
            }
        )

    if item_name:
        from ventilacia_ai.services.embeddings_service import get_text_embedding
        import numpy as np

        item_embedding = get_text_embedding(item_name)
        scored: list[tuple[float, dict[str, Any]]] = []
        for ex in correction_examples:
            ex_embedding = get_text_embedding(ex.get("original_name", ""))
            cos_sim = float(
                np.dot(item_embedding, ex_embedding)
                / (np.linalg.norm(item_embedding) * np.linalg.norm(ex_embedding) + 1e-10)
            )
            timestamp = ex.get("timestamp", "")
            recency_bonus = 0.0
            if timestamp:
                try:
                    ex_time = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                    days_ago = (datetime.now() - ex_time.replace(tzinfo=None)).days
                    recency_bonus = max(0, 0.1 * (1 - days_ago / 365))
                except Exception:
                    pass
            scored.append((cos_sim + recency_bonus, ex))
        scored.sort(key=lambda x: -x[0])
        return [ex for _, ex in scored[:limit]]

    correction_examples.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
    return correction_examples[:limit]


def confirm_results_batch(results: list[dict[str, Any]], min_confidence: float = 0.8) -> int:
    """
    Сохраняет успешные результаты сопоставления как подтверждённые коррекции.
    Вызывается при скачивании файла — пользователь принял результаты.
    Возвращает количество сохранённых записей.
    """
    training_data = load_training_data()
    existing_norms: set[str] = {
        normalize_name(c.get("original_name", ""))
        for c in training_data["corrections"]
    }

    now = datetime.now().isoformat()
    added = 0

    for r in results:
        confidence = r.get("confidence", 0)
        code = r.get("matched_code")
        name = r.get("matched_name")
        original = r.get("original_name", "")

        if not code or not original or confidence < min_confidence:
            continue

        orig_norm = normalize_name(original)
        if orig_norm in existing_norms:
            continue

        training_data["corrections"].append({
            "original_name": original,
            "corrected_code": str(code),
            "corrected_name": name or "",
            "corrected_unit": r.get("matched_unit", ""),
            "timestamp": now,
        })
        existing_norms.add(orig_norm)
        added += 1

    if added > 0:
        training_data["corrections"] = training_data["corrections"][-_MAX_CORRECTIONS:]
        save_training_data(training_data)
        print(f"[CONFIRM] Сохранено {added} подтверждённых результатов")

    return added


def add_user_correction(
    original_name: str,
    corrected_code: str,
    corrected_name: str | None = None,
    corrected_unit: str | None = None,
) -> bool:
    """Сохраняет исправление пользователя в corrections."""
    original_name = (original_name or "").strip()
    corrected_code = (corrected_code or "").strip()
    if not original_name or not corrected_code:
        return False

    corrected_name = (corrected_name or "").strip() or None
    corrected_unit = (corrected_unit or "").strip() or None

    training_data = load_training_data()

    now = datetime.now().isoformat()
    correction_record = {
        "original_name": original_name,
        "corrected_code": corrected_code,
        "corrected_name": corrected_name or "",
        "corrected_unit": corrected_unit or "",
        "timestamp": now,
    }

    original_norm = normalize_name(original_name)

    replaced = False
    new_corrections: list[dict[str, Any]] = []
    for c in training_data["corrections"]:
        c_name = str(c.get("original_name", "")).strip()
        if c_name and normalize_name(c_name) == original_norm:
            new_corrections.append(correction_record)
            replaced = True
        else:
            new_corrections.append(c)
    if not replaced:
        new_corrections.append(correction_record)
    training_data["corrections"] = new_corrections[-_MAX_CORRECTIONS:]

    return save_training_data(training_data)
