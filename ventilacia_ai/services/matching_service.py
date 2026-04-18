import json
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd

from ventilacia_ai.clients.gigachat_client import (
    GigaChatQuotaExceeded,
    get_gigachat_client,
    is_quota_exceeded,
)
from ventilacia_ai.services.embeddings_service import (
    find_top_k_candidates,
    get_text_embedding,
    nomenclature_index_ready,
    prefill_embeddings_cache,
)
from ventilacia_ai.services.text_utils import (
    dimensions_match,
    extract_dimensions,
    extract_product_type,
    normalize_name,
    product_types_compatible,
    rank_candidates,
)
from ventilacia_ai.services.training_store import (
    get_training_examples_for_prompt,
    load_training_data,
)


# ---------------------------------------------------------------------------
# 1. Поиск по пользовательским коррекциям
# ---------------------------------------------------------------------------

def find_correction(item_name: str, training_data: dict[str, Any]) -> dict[str, Any] | None:
    """Ищет исправление для указанного наименования (точное или похожее)."""
    corrections = training_data.get("corrections", [])
    if not corrections:
        return None

    item_name_normalized = normalize_name(item_name)

    # Точное совпадение после нормализации
    for correction in corrections:
        correction_name_normalized = normalize_name(correction.get("original_name", ""))
        if correction_name_normalized == item_name_normalized:
            print(f"[CORRECTION] ✅ Точное: '{item_name}'")
            return correction

    # Нечёткий поиск: embedding + text + dimension matching
    item_embedding = get_text_embedding(item_name)
    item_dims = extract_dimensions(item_name_normalized)
    best_correction = None
    best_similarity = 0.0

    for correction in corrections:
        correction_name = correction.get("original_name", "")
        correction_name_normalized = normalize_name(correction_name)
        correction_embedding = get_text_embedding(correction_name)

        cos_sim = float(
            np.dot(item_embedding, correction_embedding)
            / (np.linalg.norm(item_embedding) * np.linalg.norm(correction_embedding) + 1e-10)
        )
        text_sim = SequenceMatcher(None, item_name_normalized, correction_name_normalized).ratio()

        # Проверка числовых параметров — критически важно для вентиляционного оборудования
        corr_dims = extract_dimensions(correction_name_normalized)
        dim_score = dimensions_match(item_dims, corr_dims)

        # Если размеры не совпадают — не считаем совпадением, даже если текст похож
        if dim_score < 0.8 and item_dims:
            continue

        combined = cos_sim * 0.5 + text_sim * 0.3 + dim_score * 0.2

        if combined >= 0.88 and text_sim >= 0.85 and combined > best_similarity:
            best_similarity = combined
            best_correction = correction

    if best_correction:
        print(
            f"[CORRECTION] ✅ Нечёткое (score={best_similarity:.2f}): "
            f"'{item_name}' ~= '{best_correction.get('original_name', '')}'"
        )
        return best_correction

    return None


# ---------------------------------------------------------------------------
# 2. Точное / нормализованное / keyword совпадение с ранжированием
# ---------------------------------------------------------------------------

def _filter_by_unit(df: pd.DataFrame, source_unit: str) -> pd.DataFrame:
    """Фильтрует DataFrame по ед.изм из заявки, если она указана."""
    if not source_unit:
        return df
    unit_lower = source_unit.lower().strip().rstrip(".")
    # Нормализуем типичные варианты
    unit_map = {
        "шт": "шт", "штук": "шт", "штука": "шт",
        "пог.м": "пог.м", "п.м": "пог.м", "м.п": "пог.м", "пм": "пог.м", "погонный метр": "пог.м",
        "м": "м", "метр": "м",
        "м2": "м2", "кв.м": "м2", "м.кв": "м2",
        "кг": "кг", "килограмм": "кг",
        "компл": "компл", "комплект": "компл",
    }
    normalized_unit = unit_map.get(unit_lower, unit_lower)
    # Пытаемся отфильтровать; если ничего не осталось — возвращаем оригинал
    mask = df["Ед.изм"].astype(str).str.lower().str.strip().str.rstrip(".").map(
        lambda u: unit_map.get(u, u)
    ) == normalized_unit
    filtered = df[mask]
    return filtered if not filtered.empty else df


def find_exact_match(
    item_name: str,
    nomenclature_df: pd.DataFrame,
    source_unit: str = "",
) -> dict[str, Any] | None:
    """Пытается найти точное или частичное совпадение в номенклатуре."""
    item_lower = item_name.lower().strip()
    item_normalized = normalize_name(item_name)

    # 2a. Точное совпадение по lowercase
    exact_mask = nomenclature_df["name_lower"] == item_lower
    exact_df = nomenclature_df[exact_mask]
    if not exact_df.empty:
        exact_df = _filter_by_unit(exact_df, source_unit)
        row = exact_df.iloc[0]
        return _row_to_match(row, 1.0, "Точное совпадение")

    # 2b. Совпадение по нормализованному имени
    norm_mask = nomenclature_df["name_normalized"] == item_normalized
    norm_df = nomenclature_df[norm_mask]
    if not norm_df.empty:
        norm_df = _filter_by_unit(norm_df, source_unit)
        row = norm_df.iloc[0]
        return _row_to_match(row, 0.95, "Совпадение с учётом нормализации")

    # 2c. Keyword AND + ранжирование лучшего кандидата
    words = item_normalized.split()
    if not words:
        return None

    meaningful_words = [w for w in words if len(w) > 2]
    if not meaningful_words:
        meaningful_words = words

    mask = nomenclature_df["name_normalized"].str.contains(
        meaningful_words[0], na=False, regex=False
    )
    for word in meaningful_words[1:]:
        mask = mask & nomenclature_df["name_normalized"].str.contains(
            word, na=False, regex=False
        )

    candidates = nomenclature_df[mask]
    if candidates.empty:
        return None

    candidates = _filter_by_unit(candidates, source_unit)

    ranked = rank_candidates(item_name, candidates, top_k=1)
    if not ranked:
        return None

    best = ranked[0]
    q_dims = extract_dimensions(item_normalized)
    c_dims = extract_dimensions(normalize_name(best["name"]))
    dim_score = dimensions_match(q_dims, c_dims)

    if q_dims and dim_score < 0.5:
        return None

    confidence = min(0.80, 0.60 + best["score"] * 0.25)
    return {
        "code": best["code"],
        "name": best["name"],
        "unit": best["unit"],
        "confidence": confidence,
        "reason": "Частичное совпадение по ключевым словам (с ранжированием)",
    }


# ---------------------------------------------------------------------------
# 3. Семантический поиск — Top-K кандидатов через embedding index
# ---------------------------------------------------------------------------

def find_semantic_candidates(
    item_name: str,
    nomenclature_df: pd.DataFrame,
    top_k: int = 30,
) -> pd.DataFrame:
    """
    Находит top_k ближайших кандидатов из номенклатуры через embedding cosine similarity.
    Если индекс не готов — фоллбэк на keyword-поиск.
    """
    if nomenclature_index_ready():
        return find_top_k_candidates(item_name, nomenclature_df, top_k=top_k)

    # Фоллбэк: keyword-поиск с AND
    item_normalized = normalize_name(item_name)
    words = [w for w in item_normalized.split() if len(w) > 2]
    if not words:
        return nomenclature_df.head(top_k)

    mask = nomenclature_df["name_normalized"].str.contains(words[0], na=False, regex=False)
    for w in words[1:]:
        new_mask = mask & nomenclature_df["name_normalized"].str.contains(w, na=False, regex=False)
        if new_mask.any():
            mask = new_mask

    result = nomenclature_df[mask]
    if result.empty:
        # OR-фоллбэк если AND ничего не нашёл
        mask = pd.Series(False, index=nomenclature_df.index)
        for w in words:
            mask = mask | nomenclature_df["name_normalized"].str.contains(w, na=False, regex=False)
        result = nomenclature_df[mask]

    return result.head(top_k) if not result.empty else nomenclature_df.head(top_k)


# ---------------------------------------------------------------------------
# 4. LLM-сопоставление через GigaChat с релевантными кандидатами
# ---------------------------------------------------------------------------

def _build_prompt(
    items_for_ai: list[dict[str, Any]],
    all_candidates: list[dict[str, Any]],
    training_examples: list[dict[str, Any]],
) -> str:
    """Формирует промпт для GigaChat с релевантными кандидатами вместо случайной выборки."""

    items_lines = []
    for i, item in enumerate(items_for_ai):
        line = f"{i+1}. {item['name']}"
        if item.get("unit"):
            line += f" [ед.изм: {item['unit']}]"
        items_lines.append(line)
    items_text = "\n".join(items_lines)

    # Дедуплицируем кандидатов по коду
    seen_codes: set[str] = set()
    unique_candidates: list[dict[str, Any]] = []
    for c in all_candidates:
        code = str(c.get("Код", ""))
        if code and code not in seen_codes:
            seen_codes.add(code)
            unique_candidates.append(c)
    candidates_json = json.dumps(unique_candidates[:200], ensure_ascii=False, indent=2)

    training_text = ""
    if training_examples:
        training_text = (
            f"\n\n✅ ПРИМЕРЫ ИСПРАВЛЕНИЙ ПОЛЬЗОВАТЕЛЯ ({len(training_examples)} примеров):\n"
            "ВАЖНО: Эти примеры на 100% корректны. Используй их как эталон.\n\n"
        )
        for ex in training_examples[:50]:
            training_text += (
                f'- "{ex["original_name"]}" → Код: {ex["matched_code"]}, '
                f'Наименование: {ex["matched_name"]}'
            )
            if ex.get("matched_unit"):
                training_text += f', Ед.изм: {ex["matched_unit"]}'
            training_text += "\n"

    return f"""Ты помощник для сопоставления позиций из заявки с номенклатурой вентиляционного оборудования.
Всегда отвечай только валидным JSON массивом без дополнительных комментариев.

Номенклатура: колонки «Код», «Наименование полное», «Ед.изм».

ЗАДАЧА: для КАЖДОЙ позиции из заявки найди ОДНУ наиболее подходящую позицию из предложенных кандидатов номенклатуры.

Позиции из заявки:
{items_text}
{training_text}

КАНДИДАТЫ ИЗ НОМЕНКЛАТУРЫ (выбирай ТОЛЬКО из них!):
{candidates_json}

ПРАВИЛА СОПОСТАВЛЕНИЯ:
1. Код в ответе ОБЯЗАН совпадать с кодом одного из кандидатов выше. НЕ выдумывай коды.
2. Приоритет: точное совпадение размеров > тип изделия > единица измерения > материал > прочие характеристики.
3. «Ф», «ф», «Φ», «Ø», «ø» = диаметр. «ф100-3000» = диаметр 100, длина 3000. «Ф125-90-125» = отвод диаметр 125 угол 90°.
4. «б=0.5мм» = толщина стенки. Учитывай при выборе.
5. Размеры прямоугольного сечения: 900х600, 500x300 и т.д.
6. ЕДИНИЦА ИЗМЕРЕНИЯ: Если у позиции из заявки указана [ед.изм], выбирай кандидата с такой же единицей. «шт» и «пог.м» — это РАЗНЫЕ позиции!
7. КРИТИЧНО: тройник ≠ отвод ≠ переход ≠ воздуховод. НИКОГДА не подставляй «Отвод» вместо «Тройник» и наоборот. Тип изделия ДОЛЖЕН совпадать.
8. Если точного совпадения нет — выбирай ближайшее по типу И размеру. НЕ подставляй случайный код.
9. confidence: 0.95-1.0 = точное, 0.8-0.94 = очень близко, 0.6-0.79 = приблизительно, <0.6 = сомнительно.
10. Если ни один кандидат не подходит — верни code: null и reason с объяснением.

Формат ответа — JSON массив:
[
    {{
        "item_name": "название позиции из заявки",
        "code": "код из номенклатуры",
        "name": "полное наименование из номенклатуры",
        "unit": "единица измерения",
        "confidence": 0.95,
        "reason": "краткое объяснение выбора"
    }}
]"""


def _parse_llm_response(result_text: str) -> list[dict[str, Any]]:
    """Парсит JSON из ответа LLM с обработкой markdown-обёрток."""
    import re

    text = result_text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        json_match = re.search(r"\[[\s\S]*\]", text)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
    return []


def _verify_and_enrich(
    match_result: dict[str, Any],
    item_name: str,
    nomenclature_df: pd.DataFrame,
    training_examples: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Верифицирует результат LLM:
    - проверяет существование кода в номенклатуре
    - проверяет совпадение числовых параметров
    - при провале — ищет через семантический поиск + ранжирование
    """
    code = match_result.get("code")
    confidence = match_result.get("confidence", 0)

    item_norm = normalize_name(item_name)
    query_type = extract_product_type(item_norm)

    # Шаг 1: Если есть код — верифицируем по номенклатуре
    if code:
        verification = nomenclature_df[nomenclature_df["Код"].astype(str) == str(code)]
        if not verification.empty:
            row = verification.iloc[0]
            match_result["name"] = str(row["Наименование полное"])
            match_result["unit"] = (
                str(row["Ед.изм"]) if pd.notna(row["Ед.изм"]) else match_result.get("unit")
            )

            cand_norm = normalize_name(match_result["name"])
            cand_type = extract_product_type(cand_norm)

            # Проверка типа изделия: тройник ≠ отвод ≠ переход
            if not product_types_compatible(query_type, cand_type):
                print(
                    f"[VERIFY] ⚠️ Тип не совпадает: запрос='{query_type}', "
                    f"кандидат='{cand_type}' для '{item_name}'"
                )
                match_result["code"] = None
                match_result["confidence"] = max(0.2, confidence - 0.4)
                match_result["reason"] = (
                    f"Тип изделия не совпадает ({query_type} ≠ {cand_type}). "
                    + match_result.get("reason", "")
                )
            else:
                q_dims = extract_dimensions(item_norm)
                c_dims = extract_dimensions(cand_norm)
                dim_score = dimensions_match(q_dims, c_dims)

                if q_dims and dim_score < 0.5:
                    print(
                        f"[VERIFY] ⚠️ Код {code} не подходит по размерам "
                        f"(dim_score={dim_score:.2f}) для '{item_name}'"
                    )
                    match_result["code"] = None
                    match_result["confidence"] = max(0.3, confidence - 0.3)
                    match_result["reason"] = (
                        f"Размеры не совпали (score={dim_score:.2f}). "
                        + match_result.get("reason", "")
                    )
                elif dim_score >= 0.8:
                    match_result["confidence"] = min(1.0, confidence + 0.05)
        else:
            print(f"[VERIFY] ⚠️ Код {code} не найден в номенклатуре")
            match_result["code"] = None
            match_result["confidence"] = max(0.3, confidence - 0.3)
            match_result["reason"] = "Код не найден в номенклатуре. " + match_result.get("reason", "")

    # Шаг 2: Если код отсутствует — пытаемся найти через семантический поиск + ранжирование
    if not match_result.get("code"):
        candidates = find_semantic_candidates(item_name, nomenclature_df, top_k=50)
        if not candidates.empty:
            ranked = rank_candidates(item_name, candidates, top_k=3)
            if ranked and ranked[0]["score"] >= 0.5:
                best = ranked[0]
                q_dims = extract_dimensions(normalize_name(item_name))
                c_dims = extract_dimensions(normalize_name(best["name"]))
                dim_score = dimensions_match(q_dims, c_dims)

                # Принимаем только если размеры хотя бы частично совпадают
                if not q_dims or dim_score >= 0.5:
                    match_result["code"] = best["code"]
                    match_result["name"] = best["name"]
                    match_result["unit"] = best["unit"]
                    match_result["confidence"] = min(0.75, best["score"])
                    match_result["reason"] = (
                        f"Найдено через семантический поиск (score={best['score']:.2f}). "
                        + match_result.get("reason", "")
                    )

    # Шаг 3: Бонус за подтверждение через training examples
    if match_result.get("code") and training_examples:
        item_norm = normalize_name(item_name)
        same_code_confirmed = sum(
            1 for ex in training_examples
            if ex.get("matched_code") == match_result["code"]
            and SequenceMatcher(
                None, item_norm, normalize_name(ex.get("original_name", ""))
            ).ratio() > 0.7
        )
        if same_code_confirmed >= 1:
            match_result["confidence"] = min(0.95, match_result.get("confidence", 0) + 0.1)
            match_result["reason"] = (
                match_result.get("reason", "")
                + f" (подтверждено {same_code_confirmed} исправлениями)"
            )

    return match_result


# ---------------------------------------------------------------------------
# 5. Главная функция сопоставления
# ---------------------------------------------------------------------------

_last_run_warnings: list[str] = []


def get_last_run_warnings() -> list[str]:
    """Возвращает копию предупреждений последнего вызова `find_matching_items`."""
    return list(_last_run_warnings)


def find_matching_items(
    user_items: list[dict[str, Any]],
    nomenclature_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Высокоуровневая функция сопоставления списка позиций с номенклатурой."""
    # Сбрасываем предупреждения в начале каждого запуска.
    _last_run_warnings.clear()

    client = get_gigachat_client()
    training_data = load_training_data()

    # Предзаполняем кэш эмбеддингов
    correction_texts = [
        str(c.get("original_name", "")).strip()
        for c in training_data.get("corrections", [])
        if c.get("original_name")
    ]
    item_texts = [
        str(u.get("name", "")).strip()
        for u in user_items
        if u.get("name", "").strip()
    ]
    prefill_embeddings_cache(correction_texts + item_texts)

    results: list[dict[str, Any]] = []
    items_for_ai: list[dict[str, Any]] = []

    # ---- Этап 1: коррекции и точные совпадения ----
    for user_item in user_items:
        item_name = user_item.get("name", "").strip()
        if not item_name:
            continue

        source_unit = user_item.get("unit", "").strip()

        # 1a. Поиск по пользовательским коррекциям
        correction = find_correction(item_name, training_data)
        if correction:
            corrected_code = correction.get("corrected_code")
            if corrected_code:
                verification = nomenclature_df[
                    nomenclature_df["Код"].astype(str) == str(corrected_code)
                ]
                if not verification.empty:
                    row = verification.iloc[0]
                    is_exact = normalize_name(correction.get("original_name", "")) == normalize_name(item_name)
                    reason_detail = "точное совпадение" if is_exact else "нечёткое совпадение (≥88%)"
                    result = {
                        "original_name": item_name,
                        "source_unit": source_unit,
                        "quantity": user_item.get("quantity", ""),
                        "matched_code": corrected_code,
                        "matched_name": str(row["Наименование полное"]),
                        "matched_unit": (
                            str(row["Ед.изм"]) if pd.notna(row["Ед.изм"])
                            else correction.get("corrected_unit")
                        ),
                        "confidence": 1.0,
                        "reason": (
                            f"✅ Исправление пользователя ({reason_detail}) "
                            f"от {correction.get('timestamp', '')[:10]}"
                        ),
                    }
                    results.append(result)
                    continue

        # 1b. Точное/нормализованное/keyword совпадение
        exact_match = find_exact_match(item_name, nomenclature_df, source_unit=source_unit)
        if exact_match:
            result = {
                "original_name": item_name,
                "source_unit": source_unit,
                "quantity": user_item.get("quantity", ""),
                "matched_code": exact_match["code"],
                "matched_name": exact_match["name"],
                "matched_unit": exact_match["unit"],
                "confidence": exact_match["confidence"],
                "reason": exact_match["reason"],
            }
            results.append(result)
            continue

        items_for_ai.append(user_item)

    # ---- Этап 2: LLM-сопоставление для оставшихся позиций ----
    if items_for_ai:
        ai_results = _match_with_llm(
            client, items_for_ai, nomenclature_df, training_data
        )
        results.extend(ai_results)

    return results


def _match_with_llm(
    client,
    items_for_ai: list[dict[str, Any]],
    nomenclature_df: pd.DataFrame,
    training_data: dict[str, Any],
) -> list[dict[str, Any]]:
    """Сопоставляет позиции через GigaChat с семантическим поиском кандидатов."""

    training_examples = get_training_examples_for_prompt(limit=100)

    # Собираем релевантных кандидатов для ВСЕХ позиций
    all_candidates_dfs: list[pd.DataFrame] = []
    for user_item in items_for_ai:
        item_name = user_item.get("name", "").strip()
        if item_name:
            candidates = find_semantic_candidates(item_name, nomenclature_df, top_k=30)
            all_candidates_dfs.append(candidates)

    if all_candidates_dfs:
        merged = pd.concat(all_candidates_dfs).drop_duplicates(subset=["Код"])
        all_candidates_records = merged[
            ["Код", "Наименование полное", "Ед.изм"]
        ].to_dict("records")
    else:
        all_candidates_records = nomenclature_df.head(50)[
            ["Код", "Наименование полное", "Ед.изм"]
        ].to_dict("records")

    items_list = [
        {
            "name": u.get("name", "").strip(),
            "quantity": u.get("quantity", ""),
            "unit": u.get("unit", ""),
        }
        for u in items_for_ai
        if u.get("name", "").strip()
    ]
    if not items_list:
        return []

    prompt = _build_prompt(items_list, all_candidates_records, training_examples)

    try:
        response = client.chat(prompt)
        if hasattr(response, "choices") and len(response.choices) > 0:
            result_text = response.choices[0].message.content.strip()
        elif hasattr(response, "content"):
            result_text = response.content.strip()
        else:
            result_text = str(response).strip()
    except Exception as api_error:
        return _handle_api_error(api_error, items_for_ai, nomenclature_df)

    batch_results = _parse_llm_response(result_text)

    # Обработка результатов
    results: list[dict[str, Any]] = []
    for i, user_item in enumerate(items_for_ai):
        item_name = user_item.get("name", "").strip()
        if not item_name:
            continue

        # Сопоставляем результат по item_name или по индексу
        match_result = None
        for br in batch_results:
            if br.get("item_name") == item_name:
                match_result = br
                break
        if not match_result and i < len(batch_results):
            match_result = batch_results[i]
        if not match_result:
            match_result = {
                "code": None,
                "name": None,
                "unit": None,
                "confidence": 0,
                "reason": "Не найден в ответе нейросети",
            }

        # Верификация и обогащение
        match_result = _verify_and_enrich(
            match_result, item_name, nomenclature_df, training_examples
        )

        results.append(
            {
                "original_name": item_name,
                "source_unit": user_item.get("unit", ""),
                "quantity": user_item.get("quantity", ""),
                "matched_code": match_result.get("code"),
                "matched_name": match_result.get("name"),
                "matched_unit": match_result.get("unit"),
                "confidence": match_result.get("confidence", 0),
                "reason": match_result.get(
                    "reason",
                    "Не найдено точного соответствия. Рекомендуется исправить вручную.",
                ),
            }
        )

    return results


def _handle_api_error(
    api_error: Exception,
    items_for_ai: list[dict[str, Any]],
    nomenclature_df: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Обработка ошибок API (402 и др.) с фоллбэком на локальный поиск."""
    if not is_quota_exceeded(api_error):
        raise api_error

    print(
        "[GIGACHAT ERROR] Лимит GigaChat исчерпан (HTTP 402). "
        "Продолжаю в режиме локального поиска (ranker + embeddings)."
    )
    # Запоминаем для ответа API, чтобы фронт мог показать баннер.
    if "gigachat_quota_exceeded" not in _last_run_warnings:
        _last_run_warnings.append("gigachat_quota_exceeded")
    results: list[dict[str, Any]] = []

    for user_item in items_for_ai:
        item_name = user_item.get("name", "").strip()
        if not item_name:
            continue

        source_unit = user_item.get("unit", "")
        fallback_match = find_exact_match(item_name, nomenclature_df, source_unit=source_unit)
        if not fallback_match:
            candidates = find_semantic_candidates(item_name, nomenclature_df, top_k=50)
            if not candidates.empty:
                candidates = _filter_by_unit(candidates, source_unit)
                ranked = rank_candidates(item_name, candidates, top_k=1)
                if ranked and ranked[0]["score"] >= 0.5:
                    best = ranked[0]
                    fallback_match = {
                        "code": best["code"],
                        "name": best["name"],
                        "unit": best["unit"],
                        "confidence": min(0.75, best["score"]),
                        "reason": "Семантический поиск (GigaChat недоступен)",
                    }

        if fallback_match:
            results.append(
                {
                    "original_name": item_name,
                    "source_unit": source_unit,
                    "quantity": user_item.get("quantity", ""),
                    "matched_code": fallback_match["code"],
                    "matched_name": fallback_match["name"],
                    "matched_unit": fallback_match["unit"],
                    "confidence": fallback_match.get("confidence", 0.7),
                    "reason": (
                        f"{fallback_match.get('reason', '')} "
                        "(GigaChat API недоступен: требуется пополнение счета)"
                    ),
                }
            )
        else:
            results.append(
                {
                    "original_name": item_name,
                    "source_unit": source_unit,
                    "quantity": user_item.get("quantity", ""),
                    "matched_code": None,
                    "matched_name": None,
                    "matched_unit": None,
                    "confidence": 0,
                    "reason": (
                        "GigaChat API недоступен. Не найдено соответствия "
                        "через локальный поиск. Исправьте вручную."
                    ),
                }
            )

    return results


def _row_to_match(row: pd.Series, confidence: float, reason: str) -> dict[str, Any]:
    return {
        "code": str(row["Код"]) if pd.notna(row["Код"]) else None,
        "name": str(row["Наименование полное"]),
        "unit": str(row["Ед.изм"]) if pd.notna(row["Ед.изм"]) else None,
        "confidence": confidence,
        "reason": reason,
    }
