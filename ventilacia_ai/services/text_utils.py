import re
from difflib import SequenceMatcher
from typing import Any

import pandas as pd

# Типы изделий вентиляционного оборудования — для проверки совместимости
PRODUCT_TYPES = [
    "тройник", "отвод", "переход", "воздуховод", "заслонка", "клапан",
    "решетка", "диффузор", "дроссель", "шумоглушитель", "зонт",
    "врезка", "заглушка", "ниппель", "муфта", "адаптер", "фланец",
]


def normalize_name(name: str) -> str:
    """
    Нормализует имя для сравнения: нижний регистр, унификация символов размеров,
    раскрытие всех обозначений диаметра (Ф/ф/φ/Ø/ø/∅/D/d).
    """
    if not name:
        return ""

    normalized = name.strip().lower()

    # --- Шаг 1: Унификация всех обозначений диаметра в единый символ «ф» ---
    # Ø (U+00D8), ø (U+00F8), ∅ (U+2205), φ/Φ (Greek) → ф
    normalized = re.sub(r"[ØøΦφ∅]", "ф", normalized)

    # --- Шаг 2: Градусы: «45°» → «45» ---
    normalized = re.sub(r"(\d+)\s*°", r"\1", normalized)

    # --- Шаг 3: Унификация разделителей между ф-размерами → «/» ---
    # ф160xф125xф160 → ф160/ф125/ф160 (до унификации х в размерах!)
    while True:
        replaced = re.sub(r"ф(\d+)\s*[xXх×]\s*ф", r"ф\1/ф", normalized)
        if replaced == normalized:
            break
        normalized = replaced
    # ф250 на ф200 → ф250/ф200
    normalized = re.sub(r"ф\s*(\d+)\s+на\s+ф\s*(\d+)", r"ф\1/ф\2", normalized)

    # --- Шаг 4: Унификация «x» в размерах прямоугольных сечений → кириллическая «х» ---
    while True:
        replaced = re.sub(r"(\d)\s*[xX×]\s*(\d)", r"\1х\2", normalized)
        if replaced == normalized:
            break
        normalized = replaced

    # --- Шаг 5: Разбор формата тройников/переходов: ф160/ф125/ф160 ---
    # Три ф-числа через «/»
    normalized = re.sub(
        r"ф\s*(\d+)\s*/\s*ф\s*(\d+)\s*/\s*ф\s*(\d+)",
        r"ф\1/ф\2/ф\3",
        normalized,
    )
    # Два ф-числа через «/»
    normalized = re.sub(
        r"ф\s*(\d+)\s*/\s*ф\s*(\d+)",
        r"ф\1/ф\2",
        normalized,
    )

    # --- Шаг 6: Раскрытие «ф» с числами ---
    # ф с тремя числами через дефис: ф125-90-125 → диаметр 125 угол 90 диаметр 125
    normalized = re.sub(
        r"ф\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)",
        r"диаметр \1 угол \2 диаметр \3",
        normalized,
    )
    # ф с двумя числами через дефис, где второе — трёхзначное+ (длина в мм):
    #   ф100-3000 → диаметр 100 длина 3000.
    # Короткие/дробные «-1,5» не трогаем — это обычно количество/толщина,
    # а не длина изделия.
    normalized = re.sub(r"ф\s*(\d+)\s*-\s*(\d{3,})\b", r"диаметр \1 длина \2", normalized)
    # Тройники/переходы: ф160/ф125/ф160 → диаметр 160/диаметр 125/диаметр 160
    normalized = re.sub(r"ф(\d+)/ф(\d+)/ф(\d+)", r"диаметр \1/диаметр \2/диаметр \3", normalized)
    normalized = re.sub(r"ф(\d+)/ф(\d+)", r"диаметр \1/диаметр \2", normalized)
    # Одиночный ф + число
    normalized = re.sub(r"ф\s*-?\s*(\d+)", r"диаметр \1", normalized)
    normalized = re.sub(r"(\d+)\s*-?\s*ф", r"\1 диаметр", normalized)
    normalized = re.sub(r"\bф\b", "диаметр", normalized)

    # --- Шаг 7: Толщина стенки ---
    normalized = re.sub(r"б\s*=\s*(\d+[.,]?\d*)\s*мм", r"толщина \1", normalized)

    # --- Шаг 8: Финальная очистка ---
    normalized = " ".join(normalized.split())
    normalized = normalized.replace(" (", "(").replace("( ", "(")
    normalized = normalized.replace(" )", ")").replace(") ", ")")
    normalized = normalized.replace(" =", "=").replace("= ", "=")
    return normalized


def extract_product_type(text: str) -> str | None:
    """Извлекает тип изделия из нормализованного текста."""
    text_lower = text.lower()
    for ptype in PRODUCT_TYPES:
        if ptype in text_lower:
            return ptype
    return None


def product_types_compatible(query_type: str | None, candidate_type: str | None) -> bool:
    """Проверяет совместимость типов изделий."""
    if query_type is None or candidate_type is None:
        return True
    return query_type == candidate_type


def _to_float_mm(value: str, unit: str) -> float:
    """Приводит размер к миллиметрам."""
    num = float(value.replace(",", "."))
    if unit.startswith("м") and not unit.startswith("мм"):
        num *= 1000
    return num


def extract_dimensions(text: str) -> dict[str, list[str]]:
    """
    Извлекает числовые параметры из нормализованного текста для точного сравнения.
    Длины приводятся к миллиметрам (строкой без дробной части, если целое);
    толщина и углы сохраняются как есть.
    """
    dims: dict[str, list[str]] = {}

    for m in re.finditer(r"диаметр\s+(\d+)", text):
        dims.setdefault("diameters", []).append(m.group(1))
    for m in re.finditer(r"угол\s+(\d+)", text):
        dims.setdefault("angles", []).append(m.group(1))

    # Явные длины по ключевому слову «длина N»: считаем в мм (как раньше).
    for m in re.finditer(r"длина\s+(\d+(?:[.,]\d+)?)", text):
        dims.setdefault("lengths_mm", []).append(_canon_num(m.group(1)))

    # Длины из «Nм» (метры) и «Nмм» (миллиметры) без ключевого слова.
    # Важно не ловить «мм» как «м»: требуем, чтобы после «м» не стояла ещё «м».
    for m in re.finditer(r"(?<![а-яa-zёА-ЯA-ZЁ])(\d+(?:[.,]\d+)?)\s*м(?![а-яa-zёА-ЯA-Zё²³0-9])", text):
        num_mm = _to_float_mm(m.group(1), "м")
        # Короткие «м» значения (до 0.5м = 500мм) игнорируем — это может быть мусор.
        if num_mm >= 500:
            dims.setdefault("lengths_mm", []).append(_canon_num_float(num_mm))

    for m in re.finditer(r"(\d+(?:[.,]\d+)?)\s*мм\b", text):
        num = float(m.group(1).replace(",", "."))
        if num >= 100:
            dims.setdefault("lengths_mm", []).append(_canon_num_float(num))
        else:
            dims.setdefault("thickness", []).append(_canon_num(m.group(1)))

    # Явная толщина с ключевым словом.
    for m in re.finditer(r"толщина\s+(\d+(?:[.,]\d+)?)", text):
        dims.setdefault("thickness", []).append(_canon_num(m.group(1)))

    for m in re.finditer(r"(\d+)х(\d+)", text):
        dims.setdefault("rect", []).append(f"{m.group(1)}х{m.group(2)}")

    # Убираем дубликаты, сохраняя порядок.
    for k, vals in dims.items():
        seen: set[str] = set()
        deduped: list[str] = []
        for v in vals:
            if v not in seen:
                seen.add(v)
                deduped.append(v)
        dims[k] = deduped

    return dims


def _canon_num(raw: str) -> str:
    """Приводит '1,5' → '1.5', '3.0' → '3'."""
    num = float(raw.replace(",", "."))
    return _canon_num_float(num)


def _canon_num_float(num: float) -> str:
    if num == int(num):
        return str(int(num))
    return f"{num:g}"


def dimensions_match(query_dims: dict, candidate_dims: dict) -> float:
    """
    Оценивает, насколько числовые параметры кандидата совпадают с запросом.
    Использует мультимножественное сравнение: каждое значение кандидата
    расходуется при совпадении, чтобы ф160/ф160/ф160 ≠ ф160/ф100/ф125.
    Дополнительно штрафует за «лишнюю» длину/габариты у кандидата,
    которых нет в запросе (например, 10-метровый рулон для запроса без длины).
    Возвращает score от 0.0 до 1.0.
    """
    if not query_dims:
        return 0.5
    total = 0
    matched = 0
    for key, q_vals in query_dims.items():
        c_vals_remaining = list(candidate_dims.get(key, []))
        for v in q_vals:
            total += 1
            if v in c_vals_remaining:
                matched += 1
                c_vals_remaining.remove(v)
    if total == 0:
        return 0.5

    base = matched / total

    # Штраф за «лишние» ключевые параметры у кандидата, которых нет в запросе.
    # Длина и прямоугольное сечение — критичные: разница означает совсем другой товар.
    penalty = 0.0
    for key in ("lengths_mm", "rect"):
        q_has = bool(query_dims.get(key))
        c_has = bool(candidate_dims.get(key))
        if c_has and not q_has:
            penalty += 0.35
    return max(0.0, base - penalty)


def rank_candidates(
    query_name: str,
    candidates_df: pd.DataFrame,
    top_k: int = 1,
) -> list[dict[str, Any]]:
    """
    Ранжирует кандидатов из DataFrame по сочетанию типа изделия,
    текстового сходства и совпадения числовых параметров.
    """
    query_normalized = normalize_name(query_name)
    query_dims = extract_dimensions(query_normalized)
    query_type = extract_product_type(query_normalized)

    scored: list[tuple[float, int]] = []
    for idx, row in candidates_df.iterrows():
        cand_norm = row.get("name_normalized") or normalize_name(
            str(row["Наименование полное"])
        )

        # Штраф за несовпадение типа изделия
        cand_type = extract_product_type(cand_norm)
        if not product_types_compatible(query_type, cand_type):
            scored.append((-1.0, idx))
            continue

        text_sim = SequenceMatcher(None, query_normalized, cand_norm).ratio()
        cand_dims = extract_dimensions(cand_norm)
        dim_score = dimensions_match(query_dims, cand_dims)

        # Бонус за совпадение типа
        type_bonus = 0.1 if query_type and query_type == cand_type else 0.0

        combined = text_sim * 0.4 + dim_score * 0.5 + type_bonus
        scored.append((combined, idx))

    scored.sort(key=lambda x: -x[0])

    results = []
    for score, idx in scored[:top_k]:
        if score < 0:
            continue
        row = candidates_df.loc[idx]
        results.append(
            {
                "code": str(row["Код"]) if pd.notna(row["Код"]) else None,
                "name": str(row["Наименование полное"]),
                "unit": str(row["Ед.изм"]) if pd.notna(row["Ед.изм"]) else None,
                "score": score,
            }
        )
    return results
