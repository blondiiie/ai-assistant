"""Токенайзер для чанкования и контроля контекстного бюджета.

Использует реальный токенайзер модели (Ollama `/api/tokenize`) — cl100k_base
занижает число токенов кириллицы и не соответствует Qwen, что приводило к
молчаливому переполнению num_ctx. Счётчик инъектируется; по умолчанию (юнит-
тесты, без сети) — офлайн-эвристика.

HTTP-вызовы к Ollama делаются асинхронно (`acount`/`atokenize`), чтобы не
блокировать event loop бота. Единый кеш заполняется и синхронным, и асинхронным
путём, поэтому дублирования запросов нет.

Этап 3.4 рефакторинга: результат `/api/tokenize` кешируется на диск
(data/token_cache.json) — это убирает round-trip при каждом старте и точку
отказа при недоступности Ollama на старте.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Protocol

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

# Кириллица в BPE-токенайзерах Qwen разбивается заметно мельче, чем латиница.
_RU_CHARS_PER_TOKEN = 2.7
_LAT_CHARS_PER_TOKEN = 4.0
_CACHE_MAX = 20000

_cache: dict[tuple[str, str, str], tuple[object, ...]] = {}
_cache_lock = threading.Lock()


class TokenCounter(Protocol):
    def count(self, text: str) -> int: ...
    def tokenize(self, text: str) -> list[str] | None: ...


def _heuristic_count(text: str) -> int:
    if not text:
        return 0
    ru = sum(1 for ch in text if "\u0400" <= ch <= "\u04ff")
    other = len(text) - ru
    return max(1, int(round(ru / _RU_CHARS_PER_TOKEN + other / _LAT_CHARS_PER_TOKEN)))


# ---------------------------------------------------------------------------
# Этап 3.4: дисковый кеш токенайзера
# ---------------------------------------------------------------------------


def _disk_cache_path() -> Path:
    return Path(settings.token_cache_path)


def _disk_cache_load() -> dict[str, list]:
    """Загружает дисковый кеш в память. Пусто при ошибке."""
    if not settings.token_cache_enabled:
        return {}
    path = _disk_cache_path()
    if not path.exists():
        return {}
    try:
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            logger.info("tokens: загружен дисковый кеш (%d записей) из %s", len(data), path)
            return data
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("tokens: не удалось прочитать дисковый кеш %s: %s", path, exc)
    return {}


def _disk_cache_save(data: dict[str, list]) -> None:
    """Атомарно сохраняет дисковый кеш (write+rename)."""
    if not settings.token_cache_enabled:
        return
    path = _disk_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, path)
    except OSError as exc:
        logger.warning("tokens: не удалось сохранить дисковый кеш %s: %s", path, exc)


def _cache_key(base_url: str, model: str, text: str) -> str:
    """Строковый ключ для JSON-сериализации дискового кеша."""
    return f"{base_url}|{model}|{text}"


_disk_cache: dict[str, list] = _disk_cache_load()
_disk_lock = threading.Lock()


# ---------------------------------------------------------------------------
# In-memory кеш + общая логика
# ---------------------------------------------------------------------------


def _cache_get(key: tuple[str, str, str]) -> tuple[object, ...] | None:
    with _cache_lock:
        return _cache.get(key)


def _cache_set(key: tuple[str, str, str], val: tuple[object, ...]) -> None:
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[key] = val
    # Этап 3.4: дублируем в дисковый кеш для следующего старта.
    str_key = _cache_key(*key)
    with _disk_lock:
        _disk_cache[str_key] = list(val)
        _disk_cache_save(_disk_cache)


def _fetch_sync(base_url: str, model: str, text: str) -> tuple[object, ...]:
    key = (base_url, model, text)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    # Этап 3.4: проверяем дисковый кеш (оффлайн-старт без round-trip).
    if settings.token_cache_enabled:
        str_key = _cache_key(*key)
        with _disk_lock:
            disk_val = _disk_cache.get(str_key)
        if disk_val is not None:
            tokens = tuple(disk_val)
            with _cache_lock:
                _cache[key] = tokens
            return tokens
    with httpx.Client(timeout=settings.embed_timeout) as client:
        resp = client.post(f"{base_url}/api/tokenize", json={"model": model, "text": text})
    resp.raise_for_status()
    tokens = tuple(resp.json().get("tokens") or [])
    _cache_set(key, tokens)
    return tokens


async def _fetch_async(base_url: str, model: str, text: str) -> tuple[object, ...]:
    key = (base_url, model, text)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    if settings.token_cache_enabled:
        str_key = _cache_key(*key)
        with _disk_lock:
            disk_val = _disk_cache.get(str_key)
        if disk_val is not None:
            tokens = tuple(disk_val)
            with _cache_lock:
                _cache[key] = tokens
            return tokens
    async with httpx.AsyncClient(timeout=settings.embed_timeout) as client:
        resp = await client.post(
            f"{base_url}/api/tokenize", json={"model": model, "text": text}
        )
    resp.raise_for_status()
    tokens = tuple(resp.json().get("tokens") or [])
    _cache_set(key, tokens)
    return tokens


def _as_string_pieces(tokens: tuple[object, ...]) -> list[str] | None:
    """Куски-токены как строки, если Ollama отдал строки, иначе None.

    Современный Ollama возвращает токены строковыми кусками (по ним можно точно
    восстановить текст срезом окна). Если пришли int-id — точная нарезка
    невозможна; chunker переключится на разбиение по словам.
    """
    if not tokens or not all(isinstance(t, str) for t in tokens):
        return None
    return [str(t) for t in tokens]


class HeuristicCounter:
    """Офлайн-оценка. В проде подменяется реальным токенайзером."""

    def count(self, text: str) -> int:
        return _heuristic_count(text)

    def tokenize(self, text: str) -> list[str] | None:
        return None  # chunker использует разбиение по словам


class OllamaTokenCounter:
    def __init__(
        self,
        base_url: str = settings.ollama_url,
        model: str = settings.llm_model,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model

    def count(self, text: str) -> int:
        if not text:
            return 0
        try:
            return len(_fetch_sync(self.base_url, self.model, text))
        except (httpx.HTTPError, ValueError):
            return _heuristic_count(text)

    def tokenize(self, text: str) -> list[str] | None:
        if not text:
            return []
        try:
            tokens = _fetch_sync(self.base_url, self.model, text)
        except (httpx.HTTPError, ValueError):
            return None
        return _as_string_pieces(tokens)

    async def acount(self, text: str) -> int:
        if not text:
            return 0
        try:
            tokens = await _fetch_async(self.base_url, self.model, text)
        except (httpx.HTTPError, ValueError):
            return _heuristic_count(text)
        return len(tokens)

    async def atokenize(self, text: str) -> list[str] | None:
        if not text:
            return []
        try:
            tokens = await _fetch_async(self.base_url, self.model, text)
        except (httpx.HTTPError, ValueError):
            return None
        return _as_string_pieces(tokens)


_counter: TokenCounter | None = None


def get_counter() -> TokenCounter:
    return _counter if _counter is not None else HeuristicCounter()


def configure_counter(counter: TokenCounter) -> None:
    global _counter
    _counter = counter


async def acount_tokens(text: str) -> int:
    """Асинхронный подсчёт токенов (для async retrieval/ingest)."""
    counter = get_counter()
    if isinstance(counter, OllamaTokenCounter):
        return await counter.acount(text)
    return counter.count(text)


def count_tokens(text: str) -> int:
    """Синхронный подсчёт (офлайн-эвристика или прогретый кеш)."""
    return get_counter().count(text)