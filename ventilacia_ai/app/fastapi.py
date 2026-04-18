import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from ventilacia_ai.api.views.health_views import router as health_router
from ventilacia_ai.api.views.matching_views import router as matching_router
from ventilacia_ai.api.views.training_views import router as training_router
from ventilacia_ai.api.views.upload_views import router as upload_router
from ventilacia_ai.services.embeddings_service import (
    build_nomenclature_index,
    initialize_embeddings,
)
from ventilacia_ai.services.env_loader import ensure_env_loaded, get_last_diagnostic
from ventilacia_ai.services.nomenclature_service import load_nomenclature, nomenclature_df
from ventilacia_ai.services.training_store import load_training_data

_env_file = ensure_env_loaded()


app = FastAPI(title="ventilacia_ai", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event() -> None:
    print("=" * 60)
    print("🚀 Запуск FastAPI сервера сопоставления с обучением")
    print("=" * 60)

    has_creds = bool(
        os.getenv("GIGACHAT_CREDENTIALS")
        or (os.getenv("GIGACHAT_CLIENT_ID") and os.getenv("GIGACHAT_AUTH_KEY"))
    )
    if _env_file is not None:
        print(f"\n🔑 Переменные окружения загружены из: {_env_file}")
    elif has_creds:
        print(
            "\n🔑 Файл .env не найден, но ключи уже заданы в окружении процесса "
            "(например, через `setx` или `$env:GIGACHAT_CREDENTIALS=...`) — ок."
        )
        diag = get_last_diagnostic()
        if diag:
            print(f"   {diag}")
    else:
        print(
            "\n🔑 Файл .env НЕ найден и переменные окружения не заданы. "
            "Создайте '.env' рядом с app.py (UTF-8, без расширения .txt) "
            "или задайте ключи в оболочке."
        )
        diag = get_last_diagnostic()
        if diag:
            print(f"   {diag}")
    print(f"   GIGACHAT_CREDENTIALS: {'задан' if has_creds else 'ПУСТО'}")

    print("\n📊 Загрузка номенклатуры...")
    load_nomenclature()

    print("\n🎓 Инициализация системы обучения...")
    training_data = load_training_data()
    corrections_count = len(training_data.get("corrections", []))
    print(f"   Исправлений пользователя: {corrections_count}")

    print("\n🔍 Инициализация векторного поиска...")
    initialize_embeddings()

    if nomenclature_df is not None and len(nomenclature_df) > 0:
        print("\n📐 Построение индекса номенклатуры для семантического поиска...")
        build_nomenclature_index(nomenclature_df)

    print("\n" + "=" * 60)
    print("✅ FastAPI приложение готово к работе с векторным поиском!")
    print("=" * 60 + "\n")


app.include_router(health_router)
app.include_router(matching_router)
app.include_router(training_router)
app.include_router(upload_router)


@app.get("/")
async def index() -> FileResponse:
    """Отдаёт основной frontend (`index.html`) по корневому URL."""
    return FileResponse("index.html")

