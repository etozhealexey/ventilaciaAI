import os
from typing import Any, List

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ventilacia_ai.models.schemas import MatchRequest
from ventilacia_ai.services import config_service
from ventilacia_ai.services import matching_service, excel_service, nomenclature_service
from ventilacia_ai.services import training_store

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
    results = matching_service.find_matching_items(items, nomenclature_service.nomenclature_df)
    filename = excel_service.create_excel_file(results)

    return JSONResponse(
        {
            "success": True,
            "results": results,
            "filename": filename,
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

