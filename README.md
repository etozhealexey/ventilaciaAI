# Ventilacia AI

Веб‑приложение для сопоставления позиций из заявки с номенклатурой.

## Требования

- Python **3.10+**
- macOS / Linux / Windows

## Быстрый старт (локально)

1) Создайте виртуальное окружение:

```bash
cd <путь до папки с кодом>
python3 -m venv venv
```

2) Активируйте окружение:

```bash
# macOS / Linux
source venv/bin/activate

# Windows (PowerShell)
# .\\venv\\Scripts\\Activate.ps1
```

3) Установите зависимости:

```bash
pip install -U pip
pip install -r requirements.txt
```

4) Настройте переменные окружения:

```bash
cp .env.example .env
```

Переменные для .env запросить у администратора

5) Убедитесь, что в корне проекта есть файлы:

- `номенклатура.csv`
- `training_data.json` (если нет — создайте пустой файл с содержимым `{"corrections": []}`)

6) Запустите приложение:

```bash
PORT=5001 python app.py
```

Откройте в браузере:

- `http://localhost:5001`


## Запуск через Docker Compose

1) Создайте `.env`:

```bash
cp .env.example .env
```

2) Запустите:

```bash
docker compose up --build
```

3) Откройте:

- `http://localhost:5000`

Docker монтирует:

- `номенклатура.csv` (read‑only)
- `uploads/`, `reports/`
- `training_data.json`
- `nomenclature_embeddings.npz` (кэш)

