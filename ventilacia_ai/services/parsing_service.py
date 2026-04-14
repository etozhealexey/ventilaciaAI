import json
import re
from typing import Any

import pandas as pd
import pdfplumber
from docx import Document

from ventilacia_ai.clients.gigachat_client import get_gigachat_client
from ventilacia_ai.services import config_service


def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in config_service.ALLOWED_EXTENSIONS
    )


def parse_excel_application(file_path: str) -> list[dict[str, Any]]:
    """
    Парсит Excel файл заявки фиксированного формата:
    - 1 столбец: наименование
    - 2 столбец: количество
    - 3 столбец: единица измерения (опционально)
    """
    items: list[dict[str, Any]] = []
    try:
        df = pd.read_excel(file_path, header=None)

        # Набор типичных заголовков для пропуска
        header_names = {
            "наименование", "название", "name", "позиция",
            "наименование позиции", "наименование товара",
        }

        for idx, row in df.iterrows():
            if len(row) < 1:
                continue

            name = str(row.iloc[0]).strip() if pd.notna(row.iloc[0]) else ""
            if not name or name.lower() in header_names:
                continue

            quantity = "1"
            if len(row) > 1 and pd.notna(row.iloc[1]):
                qty_str = str(row.iloc[1]).strip()
                m = re.search(r"(\d+(?:[.,]\d+)?)", qty_str)
                if m:
                    quantity = m.group(1).replace(",", ".")

            unit = ""
            if len(row) > 2 and pd.notna(row.iloc[2]):
                unit_raw = str(row.iloc[2]).strip()
                if unit_raw.lower() not in {"ед.изм", "ед. изм", "ед.изм.", "единица измерения", "unit", ""}:
                    unit = unit_raw

            items.append(
                {
                    "row_number": idx + 1,
                    "name": name,
                    "quantity": quantity,
                    "unit": unit,
                }
            )

        return items
    except Exception as e:
        print(f"Ошибка при парсинге Excel: {e}")
        return []


def parse_docx_application(file_path: str) -> list[dict[str, Any]]:
    """Парсит Word документ заявки и извлекает позиции."""
    try:
        doc = Document(file_path)
        structured_text = ""

        for idx, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if text and len(text) > 5:
                structured_text += f"Строка {idx + 1}: {text}\n"

        for table_idx, table in enumerate(doc.tables):
            for row_idx, row in enumerate(table.rows):
                row_data = []
                for cell in row.cells:
                    cell_text = cell.text.strip()
                    if cell_text:
                        row_data.append(cell_text)
                if row_data:
                    row_text = " | ".join(row_data)
                    if len(row_text) > 10:
                        structured_text += (
                            f"Таблица {table_idx + 1}, Строка {row_idx + 1}: {row_text}\n"
                        )

        return parse_text_with_ai(structured_text)
    except Exception as e:
        print(f"Ошибка при парсинге Word: {e}")
        return []


def parse_pdf_application(pdf_path: str) -> list[dict[str, Any]]:
    """Парсит PDF файл заявки и извлекает позиции."""
    try:
        table_rows = []
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages):
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 1:
                        for row_idx, row in enumerate(table[2:], start=3):
                            if row and len(row) > 0:
                                non_empty_cells = [
                                    str(cell).strip() if cell else ""
                                    for cell in row
                                    if cell
                                ]
                                if any(
                                    cell for cell in non_empty_cells if len(str(cell)) > 2
                                ):
                                    table_rows.append(
                                        {
                                            "page": page_num + 1,
                                            "row": row_idx,
                                            "cells": non_empty_cells,
                                        }
                                    )

        structured_text = ""
        for row_info in table_rows:
            row_text = " | ".join(row_info["cells"])
            if row_text and len(row_text) > 10:
                structured_text += f"Строка {row_info['row']}: {row_text}\n"

        full_text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    full_text += text + "\n"

        return parse_text_with_ai(structured_text if structured_text else full_text)
    except Exception as e:
        print(f"Ошибка при парсинге PDF: {e}")
        return []


def parse_text_with_ai(text: str) -> list[dict[str, Any]]:
    """Использует GigaChat для извлечения позиций из текста."""
    try:
        client = get_gigachat_client()
    except ValueError as e:
        print(f"Не удалось создать клиент GigaChat: {e}")
        return []

    try:
        prompt = f"""Ты помощник для извлечения позиций товаров/материалов из комплектовочной ведомости.

Из следующего текста извлеки ТОЛЬКО позиции товаров/материалов из таблицы.

ВАЖНО: Игнорируй полностью:
- Заголовки документа
- Имена людей
- Даты
- Заголовки колонок таблицы
- Служебную информацию
- Пустые строки и разделители
- Искаженный/нечитаемый текст

Структурированные данные:
{text[:8000]}

Верни JSON массив объектов в формате:
[
    {{
        "row_number": 1,
        "name": "полное наименование товара/материала с техническими характеристиками",
        "quantity": "количество (только число)"
    }}
]"""
        full_prompt = (
            "Ты помощник для извлечения позиций товаров/материалов из комплектовочных ведомостей. "
            "Всегда отвечай только валидным JSON массивом без дополнительных комментариев.\n\n"
            + prompt
        )
        response = client.chat(full_prompt)

        if hasattr(response, "choices") and len(response.choices) > 0:
            result_text = response.choices[0].message.content.strip()
        elif hasattr(response, "content"):
            result_text = response.content.strip()
        else:
            result_text = str(response).strip()

        if result_text.startswith("```json"):
            result_text = result_text[7:]
        if result_text.startswith("```"):
            result_text = result_text[3:]
        if result_text.endswith("```"):
            result_text = result_text[:-3]
        result_text = result_text.strip()

        items = json.loads(result_text)
        return validate_and_clean_items(items)
    except Exception as e:
        print(f"Ошибка при парсинге через нейросеть: {e}")
        return []


def validate_and_clean_items(items: Any) -> list[dict[str, Any]]:
    """Валидация и очистка извлеченных позиций."""
    validated_items: list[dict[str, Any]] = []

    exclude_keywords = [
        "комплектовочная",
        "ведомость",
        "материалов",
        "оборудования",
        "утверждаю",
        "начальник",
        "филиала",
        "подпись",
        "инициалы",
        "фамилия",
        "января",
        "февраля",
        "марта",
        "апреля",
        "мая",
        "июня",
        "июля",
        "августа",
        "сентября",
        "октября",
        "ноября",
        "декабря",
        "года",
        "г.",
        "наименование",
        "технические",
        "характеристики",
        "тип",
        "марка",
        "гост",
        "ту",
        "завод",
        "изготовитель",
        "ед",
        "изм",
        "кол-во",
        "стоимость",
        "смете",
        "цена",
        "ндс",
        "статья",
        "затрат",
        "месяц",
        "поставки",
        "гпр",
        "примечание",
        "не предъявлено",
        "заказчику",
        "шифр",
        "объекта",
        "строительства",
        "локальный",
        "сметный",
        "расчет",
        "отсутствует",
    ]

    def is_readable_text(text: str) -> bool:
        if not text or len(text) < 5:
            return False
        has_cyrillic = bool(re.search(r"[А-Яа-яЁё]", text))
        consonant_sequences = re.findall(
            r"[бвгджзклмнпрстфхцчшщБВГДЖЗКЛМНПРСТФХЦЧШЩ]{5,}", text
        )
        if consonant_sequences:
            return False
        words = re.findall(r"[А-Яа-яЁё]{3,}", text)
        if len(words) < 2 and has_cyrillic:
            return False
        return True

    if not isinstance(items, list):
        return []

    for item in items:
        if not isinstance(item, dict) or "name" not in item:
            continue

        name = str(item.get("name", "")).strip()
        row_number = item.get("row_number", None)

        if not name or len(name) < 5:
            continue
        if not is_readable_text(name):
            continue

        name_lower = name.lower()
        if any(keyword in name_lower for keyword in exclude_keywords):
            has_technical_info = any(char.isdigit() for char in name) and (
                "мм" in name_lower
                or "м.п" in name_lower
                or "м " in name_lower
                or "кг" in name_lower
                or "шт" in name_lower
                or "гост" in name_lower
                or "ту" in name_lower
                or len(name) > 20
            )
            if not has_technical_info:
                continue

        if re.match(r"^[А-ЯЁ]\.\s*[А-ЯЁ]\.\s*[А-ЯЁ][а-яё]+$", name):
            continue

        if re.search(
            r"\d{1,2}\s*(января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)\s*\d{4}",
            name_lower,
        ):
            continue

        name = re.sub(r"^[\|\s]+", "", name)

        quantity = str(item.get("quantity", "1")).strip() or "1"
        quantity = re.sub(r"[^\d.,]", "", quantity)
        if not quantity or quantity == "0":
            quantity = "1"

        unit = str(item.get("unit", "")).strip()

        validated_items.append(
            {
                "row_number": row_number if row_number else len(validated_items) + 1,
                "name": name,
                "quantity": quantity,
                "unit": unit,
            }
        )

    validated_items.sort(key=lambda x: x.get("row_number", 999))
    return validated_items


def parse_application_file(file_path: str, extension: str) -> list[dict[str, Any]]:
    """Единая точка входа для парсинга файла заявки (поддерживается только Excel)."""
    ext = (extension or "").lower().lstrip(".")
    if ext in {"xlsx", "xls"}:
        return parse_excel_application(file_path)
    return []

