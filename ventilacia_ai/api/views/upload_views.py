from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse

from ventilacia_ai.api.controllers.upload_controller import parse_uploaded_application

router = APIRouter()


@router.post("/api/upload-application")
async def upload_application(file: UploadFile = File(...)) -> JSONResponse:
    payload = await parse_uploaded_application(file)
    return JSONResponse(payload)

