"""
Compatibility entrypoint.

Основное приложение находится в пакете `ventilacia_ai` (`ventilacia_ai.app.fastapi:app`).
Файл оставлен только для запуска командой `python app.py`.
"""

from ventilacia_ai.app.fastapi import app  # noqa: F401


if __name__ == "__main__":
    import os

    import uvicorn

    port = int(os.getenv("PORT", "5000"))
    uvicorn.run("ventilacia_ai.app.fastapi:app", host="0.0.0.0", port=port, reload=False)
