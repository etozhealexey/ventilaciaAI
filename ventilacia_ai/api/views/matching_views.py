import os
from typing import Any, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ventilacia_ai.clients.gigachat_client import (
    GigaChatQuotaExceeded,
    is_quota_exceeded,
)
from ventilacia_ai.models.schemas import MatchRequest
from ventilacia_ai.services import config_service
from ventilacia_ai.services import matching_service, excel_service, nomenclature_service
from ventilacia_ai.services import training_store


_QUOTA_MESSAGE = (
    "Лимит GigaChat исчерпан (HTTP 402 Payment Required). "
    "Пополните баланс в личном кабинете Сбера "
    "(https://developers.sber.ru/studio/) или дождитесь сброса "
    "дневного/месячного лимита."
)

router = APIRouter()


class ConfirmResultsRequest(BaseModel):
    results: List[dict[str, Any]]


@router.post("/api/match")
async def match_items(payload: MatchRequest) -> JSONResponse:
    """Сопоставление позиций."""
    if not payload.items:
        raise HTTPException(status_code=400, detail="Список позиций пуст")

    if nomenclature_service.nomenclature_df is None:
        raise HTTPException(status_code=500, detail="Номенклатура не загружена на сервере")

    items: List[dict] = [
        {"name": i.name, "quantity": i.quantity or "", "unit": i.unit or ""}
        for i in payload.items
    ]
    try:
        results = matching_service.find_matching_items(
            items, nomenclature_service.nomenclature_df
        )
    except GigaChatQuotaExceeded:
        return JSONResponse(
            {
                "success": False,
                "error_code": "gigachat_quota_exceeded",
                "error": _QUOTA_MESSAGE,
            }
        )
    except Exception as e:
        if is_quota_exceeded(e):
            return JSONResponse(
                {
                    "success": False,
                    "error_code": "gigachat_quota_exceeded",
                    "error": _QUOTA_MESSAGE,
                    "detail": str(e),
                }
            )
        raise

    warnings = matching_service.get_last_run_warnings()
    filename = excel_service.create_excel_file(results)

    return JSONResponse(
        {
            "success": True,
            "results": results,
            "filename": filename,
            "warnings": warnings,
        }
    )


@router.post("/api/confirm-results")
async def confirm_results(payload: ConfirmResultsRequest) -> JSONResponse:
    """Сохраняет принятые пользователем результаты как обучающие данные."""
    saved = training_store.confirm_results_batch(payload.results)
    return JSONResponse({"success": True, "saved": saved})


@router.get("/api/download/{filename}")
async def download_file(filename: str) -> FileResponse:
    """Скачивание сгенерированного Excel файла."""
    filepath = os.path.join(config_service.REPORTS_FOLDER, filename)
    if not os.path.exists(filepath):
        if os.path.exists(filename):
            filepath = filename
        else:
            raise HTTPException(status_code=404, detail=f"Файл {filename} не найден")

    return FileResponse(
        filepath,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=filename,
    )

