"""
Единая точка загрузки переменных окружения из `.env`.

Предназначена для устранения Windows-ловушек:
- `.env.txt` от Блокнота (скрытое расширение),
- BOM в начале файла (UTF-8 BOM или UTF-16 с BOM),
- запуск сервера не из корня проекта (load_dotenv() без пути ничего не находит).

Функция `ensure_env_loaded()` идемпотентна — её можно вызывать многократно.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

_CANDIDATE_FILES = (
    ".env",
    ".env.txt",   # Блокнот Windows любит дописывать .txt
    ".env.local",
    ".ENV",
    ".Env",
    "env",        # совсем без точки (иногда встречается)
)

_loaded = False
_found_path: Path | None = None
_last_diagnostic: str | None = None


def get_last_diagnostic() -> str | None:
    """Возвращает последнее диагностическое сообщение (для вывода на старте)."""
    return _last_diagnostic


def _build_diagnostic(prefix: str) -> str:
    """Собирает информацию о путях, которую видит процесс."""
    try:
        entries = sorted(p.name for p in _PROJECT_ROOT.iterdir())
    except OSError:
        entries = []
    env_like = [n for n in entries if "env" in n.lower()]
    lines = [
        prefix,
        f"Корень проекта для поиска .env: {_PROJECT_ROOT}",
        f"Текущий рабочий каталог (CWD):  {Path.cwd()}",
    ]
    if env_like:
        lines.append("Файлы с 'env' в имени в корне: " + ", ".join(env_like))
    else:
        lines.append("В корне нет ни одного файла, содержащего 'env' в имени.")
    return "\n   ".join(lines)


def _read_with_fallback_encodings(path: Path) -> str | None:
    """Читает файл, корректно отбрасывая BOM (UTF-8-SIG, UTF-16)."""
    for encoding in ("utf-8-sig", "utf-16", "cp1251", "utf-8"):
        try:
            return path.read_text(encoding=encoding)
        except (UnicodeError, UnicodeDecodeError):
            continue
    return None


def _apply_env_text(text: str) -> int:
    """
    Разбирает содержимое .env вручную и выставляет переменные, которых ещё нет
    в окружении. Возвращает количество установленных переменных.
    """
    count = 0
    for raw_line in text.splitlines():
        line = raw_line.strip().lstrip("\ufeff")
        if not line or line.startswith("#"):
            continue
        if line.lower().startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if not key:
            continue
        if os.environ.get(key):
            continue
        os.environ[key] = value
        count += 1
    return count


def ensure_env_loaded() -> Path | None:
    """
    Загружает переменные окружения из `.env` рядом с корнем проекта.
    Возвращает путь к реально использованному файлу (или None, если не найден).
    Идемпотентна: при повторных вызовах возвращает тот же результат, что и первый.
    """
    global _loaded, _found_path, _last_diagnostic
    if _loaded:
        return _found_path

    found: Path | None = None
    for name in _CANDIDATE_FILES:
        candidate = _PROJECT_ROOT / name
        if candidate.is_file():
            found = candidate
            break

    if found is None:
        _last_diagnostic = _build_diagnostic(
            "Диагностика поиска .env:"
        )
        _loaded = True
        _found_path = None
        return None

    loaded_ok = False
    try:
        load_dotenv(dotenv_path=found, override=False, encoding="utf-8")
        loaded_ok = True
    except (UnicodeDecodeError, UnicodeError):
        # Скорее всего файл сохранён Блокнотом как UTF-16 — пойдём ручным путём.
        loaded_ok = False

    # Если python-dotenv не справился или ключевые переменные всё ещё пустые —
    # парсим файл вручную с подбором кодировки.
    if (
        not loaded_ok
        or (not os.getenv("GIGACHAT_CREDENTIALS") and not os.getenv("GIGACHAT_CLIENT_ID"))
    ):
        text = _read_with_fallback_encodings(found)
        if text is not None:
            _apply_env_text(text)

    _last_diagnostic = _build_diagnostic(f".env прочитан: {found}")
    _loaded = True
    _found_path = found
    return found
