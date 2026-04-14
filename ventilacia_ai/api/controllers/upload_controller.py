import os
import uuid
from typing import Any

from fastapi import UploadFile

from ventilacia_ai.services import config_service
from ventilacia_ai.services.parsing_service import parse_application_file


async def parse_uploaded_application(file: UploadFile) -> dict[str, Any]:
    """
    Сохраняет загруженный файл во временный путь, парсит позиции, затем удаляет файл.
    Возвращает структуру для ответа API.
    """
    if file is None or not file.filename:
        return {"success": False, "error": "Файл не выбран"}

    # Расширение берём из исходного имени: secure_filename() выкидывает кириллицу и
    # ломает проверку для файлов вроде «Заявка.xlsx» → остаётся «xlsx» без точки.
    raw_name = file.filename.strip()
    if "." not in raw_name:
        return {
            "success": False,
            "error": "Неподдерживаемый тип файла. Разрешены только Excel-файлы (.xlsx, .xls)",
        }
    ext = raw_name.rsplit(".", 1)[1].lower()
    if ext not in config_service.ALLOWED_EXTENSIONS:
        return {
            "success": False,
            "error": "Неподдерживаемый тип файла. Разрешены только Excel-файлы (.xlsx, .xls)",
        }

    os.makedirs(config_service.UPLOAD_FOLDER, exist_ok=True)
    temp_path = os.path.join(
        config_service.UPLOAD_FOLDER, f"upload_{uuid.uuid4().hex}.{ext}"
    )

    try:
        content = await file.read()
        with open(temp_path, "wb") as f:
            f.write(content)

        items = parse_application_file(temp_path, ext)
        items = items or []
        return {
            "success": True,
            "items": items,
            "count": len(items),
        }
    except Exception as e:
        return {"success": False, "error": f"Ошибка обработки файла: {e}"}
    finally:
        try:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            pass

