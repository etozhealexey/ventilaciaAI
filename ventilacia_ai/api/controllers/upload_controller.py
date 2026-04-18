import os
import uuid
from typing import Any

from fastapi import UploadFile

from ventilacia_ai.clients.gigachat_client import GigaChatQuotaExceeded
from ventilacia_ai.services import config_service
from ventilacia_ai.services.parsing_service import (
    parse_application_file,
    parse_application_file_smart,
)


async def _parse_upload(file: UploadFile, smart: bool) -> dict[str, Any]:
    """
    Общая логика: сохраняет загруженный файл во временный путь, парсит позиции,
    затем удаляет файл. Выбирает парсер по флагу smart.
    """
    if file is None or not file.filename:
        return {"success": False, "error": "Файл не выбран"}

    # Расширение берём из исходного имени: secure_filename() выкидывает кириллицу и
    # ломает проверку для файлов вроде «Заявка.xlsx» → остаётся «xlsx» без точки.
    raw_name = file.filename.strip()
    allowed = (
        config_service.ALLOWED_EXTENSIONS_SMART if smart else config_service.ALLOWED_EXTENSIONS
    )
    allowed_list = ", ".join(sorted(f".{e}" for e in allowed))
    if "." not in raw_name:
        return {
            "success": False,
            "error": f"Неподдерживаемый тип файла. Разрешены: {allowed_list}",
        }
    ext = raw_name.rsplit(".", 1)[1].lower()
    if ext not in allowed:
        return {
            "success": False,
            "error": f"Неподдерживаемый тип файла. Разрешены: {allowed_list}",
        }

    os.makedirs(config_service.UPLOAD_FOLDER, exist_ok=True)
    temp_path = os.path.join(
        config_service.UPLOAD_FOLDER, f"upload_{uuid.uuid4().hex}.{ext}"
    )

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        parser = parse_application_file_smart if smart else parse_application_file
        items = parser(temp_path, ext) or []
        # Перенумеровываем строго по порядку (1..N), чтобы пользователь видел
        # позиционные номера, а не номера строк исходного документа.
        for idx, item in enumerate(items, start=1):
            if isinstance(item, dict):
                item["row_number"] = idx
        payload: dict[str, Any] = {
            "success": True,
            "items": items,
            "count": len(items),
        }
        if not items:
            if ext == "pdf":
                payload["warning"] = (
                    "Позиции не найдены. Возможно, это скан PDF без текстового слоя — "
                    "такой формат пока не поддерживается."
                )
            elif ext == "docx":
                payload["warning"] = (
                    "Позиции не найдены. Проверьте, что в документе есть таблица или список позиций."
                )
            elif smart:
                payload["warning"] = "Позиции не найдены. Проверьте содержимое файла."
            else:
                payload["warning"] = (
                    "Позиции не найдены. Проверьте формат: 1 колонка — наименование, "
                    "2 — количество, 3 — ед.изм. Либо воспользуйтесь режимом «Распознать заявку»."
                )
        return payload
    except GigaChatQuotaExceeded as quota_err:
        return {
            "success": False,
            "error_code": "gigachat_quota_exceeded",
            "error": (
                "Лимит GigaChat исчерпан (HTTP 402 Payment Required). "
                "Пополните баланс в личном кабинете Сбера "
                "(https://developers.sber.ru/studio/) или дождитесь сброса "
                "дневного/месячного лимита. Попробуйте позже или воспользуйтесь "
                "режимом «По шаблону», который не использует нейросеть."
            ),
            "detail": str(quota_err),
        }
    except Exception as e:
        return {"success": False, "error": f"Ошибка обработки файла: {e}"}
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass


async def parse_uploaded_application(file: UploadFile) -> dict[str, Any]:
    """Строгий режим: только xlsx/xls фиксированного формата, без LLM."""
    return await _parse_upload(file, smart=False)


async def parse_uploaded_application_smart(file: UploadFile) -> dict[str, Any]:
    """Умный режим: xlsx/xls/docx/pdf — позиции извлекаются через GigaChat."""
    return await _parse_upload(file, smart=True)
