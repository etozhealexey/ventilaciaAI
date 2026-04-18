import os
import ssl

from ventilacia_ai.services.env_loader import ensure_env_loaded

try:
    from gigachat import GigaChat

    GIGACHAT_AVAILABLE = True
except ImportError:
    GIGACHAT_AVAILABLE = False
    print("Предупреждение: библиотека gigachat не установлена. Установите: pip install gigachat")


ensure_env_loaded()

# По умолчанию SDK подставляет модель «GigaChat» (часто это Lite). Если лимит Lite исчерпан,
# а подписка на Pro/Max активна — задайте GIGACHAT_MODEL=GigaChat-Pro или GigaChat-Max в .env.
_DEFAULT_CHAT_MODEL = "GigaChat-Pro"


def get_gigachat_client() -> "GigaChat":
    """Создает и возвращает клиент GigaChat с учётом переменных окружения и SSL."""
    if not GIGACHAT_AVAILABLE:
        raise ValueError("Библиотека gigachat не установлена. Установите: pip install gigachat")

    credentials = os.getenv("GIGACHAT_CREDENTIALS")
    if not credentials:
        client_id = os.getenv("GIGACHAT_CLIENT_ID")
        auth_key = os.getenv("GIGACHAT_AUTH_KEY")
        if client_id and auth_key:
            credentials = f"{client_id}:{auth_key}"
        else:
            raise ValueError(
                "GIGACHAT_CREDENTIALS не установлен в переменных окружения.\n"
                "Укажите либо GIGACHAT_CREDENTIALS (ключ авторизации),\n"
                "либо GIGACHAT_CLIENT_ID и GIGACHAT_AUTH_KEY отдельно."
            )

    os.environ["PYTHONHTTPSVERIFY"] = "0"
    os.environ["CURL_CA_BUNDLE"] = ""
    os.environ["REQUESTS_CA_BUNDLE"] = ""

    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE

    model = os.getenv("GIGACHAT_MODEL", _DEFAULT_CHAT_MODEL).strip() or _DEFAULT_CHAT_MODEL

    # Увеличенный тайм-аут нужен для длинных промптов (например, разбор заявок с таблицами).
    try:
        timeout_sec = float(os.getenv("GIGACHAT_TIMEOUT", "180"))
    except ValueError:
        timeout_sec = 180.0

    try:
        return GigaChat(
            credentials=credentials,
            verify_ssl_certs=False,
            model=model,
            timeout=timeout_sec,
        )
    except (TypeError, ValueError):
        try:
            return GigaChat(
                credentials=credentials,
                model=model,
                timeout=timeout_sec,
            )
        except (TypeError, ValueError):
            return GigaChat(credentials=credentials, model=model)


