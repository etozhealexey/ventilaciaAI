from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from ventilacia_ai.api.controllers.upload_controller import (
    parse_uploaded_application,
    parse_uploaded_application_smart,
)

router = APIRouter()


@router.post("/api/upload-application")
async def upload_application(file: UploadFile = File(...)) -> JSONResponse:
    """Строгий режим: xlsx/xls фиксированного формата, без LLM."""
    payload = await parse_uploaded_application(file)
    return JSONResponse(payload)


@router.post("/api/upload-application-smart")
async def upload_application_smart(file: UploadFile = File(...)) -> JSONResponse:
    """Умный режим: xlsx/xls/docx/pdf — позиции извлекаются через GigaChat."""
    payload = await parse_uploaded_application_smart(file)
    return JSONResponse(payload)
