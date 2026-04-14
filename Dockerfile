# Используем официальный Python образ
FROM python:3.11-slim

# Устанавливаем рабочую директорию
WORKDIR /app

# Устанавливаем системные зависимости
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    make \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Копируем файл зависимостей
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Устанавливаем зависимости по отдельности для лучшей диагностики
RUN pip install --no-cache-dir -r requirements.txt

# Проверяем установку критически важных библиотек
RUN python -c "import fastapi; import uvicorn; import pandas; import gigachat; print('All critical libraries installed successfully')" || \
    (echo "ERROR: Some libraries installation failed" && pip list && exit 1)

# Копируем все файлы приложения
COPY . .

# Создаем необходимые директории
RUN mkdir -p uploads reports

# Открываем порт
EXPOSE 5000

# Переменные окружения
ENV PYTHONUNBUFFERED=1

# Команда запуска FastAPI через uvicorn (используем пакет ventilacia_ai)
CMD ["uvicorn", "ventilacia_ai.app.fastapi:app", "--host", "0.0.0.0", "--port", "5000"]
