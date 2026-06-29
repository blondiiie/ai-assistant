.PHONY: up down venv install smoke test lint format profile-ram

up:  ## Поднять Postgres+pgvector
	docker compose up -d

initdb:  ## Создать схему (расширения, таблицы, индексы)
	uv run python -m app.db.init_db

reindex:  ## Полное переиндексирование (очистка + скан). Применять после правок парсера
	uv run python -m app.sync.reindex

serve:  ## Запустить FastAPI
	uv run uvicorn app.api.main:app --reload

bot:  ## Запустить Telegram-бота
	uv run python -m app.bot.main

down:  ## Остановить Postgres
	docker compose down

venv:  ## Создать виртуальное окружение (Python 3.12)
	uv venv --python 3.12

install:  ## Установить зависимости
	uv sync --extra dev

smoke:  ## Smoke-тест Ollama (Фаза 0)
	uv run python -m app.smoke_test

test:  ## Запустить тесты
	uv run pytest

lint:  ## Линтинг
	uv run ruff check .

format:  ## Форматирование
	uv run ruff format .

profile-ram:  ## Профиль RAM (Этап 4.3): 5 запросов + метрики psutil, отчёт в data/ram_profile.json
	uv run python -m scripts.profile_ram
