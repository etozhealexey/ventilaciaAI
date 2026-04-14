from fastapi import APIRouter
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse

router = APIRouter()


@router.get("/test")
async def test() -> JSONResponse:
    return JSONResponse({"status": "ok", "message": "FastAPI сервер работает!"})


@router.get("/health", response_class=PlainTextResponse)
async def health() -> PlainTextResponse:
    return PlainTextResponse("OK")


@router.get("/simple", response_class=HTMLResponse)
async def simple() -> HTMLResponse:
    html = (
        "<html><body><h1>FastAPI сервер работает!</h1>"
        "<p>Если вы видите это сообщение, значит FastAPI отвечает корректно.</p>"
        "</body></html>"
    )
    return HTMLResponse(content=html)

