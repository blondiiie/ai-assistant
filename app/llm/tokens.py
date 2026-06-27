"""Токенайзер для чанкования и контроля контекстного бюджета.

Использует реальный токенайзер модели (Ollama `/api/tokenize`) — cl100k_base
занижает число токенов кириллицы и не соответствует Qwen, что приводило к
молчаливому переполнению num_ctx. Счётчик инъектируется; по умолчанию (юнит-
тесты, без сети) — офлайн-эвристика.

HTTP-вызовы к Ollama делаются асинхронно (`acount`/`atokenize`), чтобы не
блокировать event loop бота. Единый кеш заполняется и синхронным, и асинхронным
путём, поэтому дублирования запросов нет.
"""
from __future__ import annotations

import threading
from typing import Protocol

import httpx

from app.config import settings

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


def _cache_get(key: tuple[str, str, str]) -> tuple[object, ...] | None:
    with _cache_lock:
        return _cache.get(key)


def _cache_set(key: tuple[str, str, str], val: tuple[object, ...]) -> None:
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[key] = val


def _fetch_sync(base_url: str, model: str, text: str) -> tuple[object, ...]:
    key = (base_url, model, text)
    cached = _cache_get(key)
    if cached is not None:
        return cached
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
