from __future__ import annotations

import asyncio
from typing import Any

import httpx

from app.config import settings


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(
        self,
        base_url: str = settings.ollama_url,
        embed_model: str = settings.embed_model,
        chat_model: str = settings.llm_model,
        embed_timeout: float = settings.embed_timeout,
        chat_timeout: float = settings.llm_timeout,
        max_concurrent: int = settings.max_concurrent_llm,
        num_ctx: int = settings.llm_num_ctx,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.embed_model = embed_model
        self.chat_model = chat_model
        self.embed_timeout = embed_timeout
        self.chat_timeout = chat_timeout
        self.num_ctx = num_ctx
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        async with httpx.AsyncClient(timeout=self.embed_timeout) as client:
            resp = await client.post(
                f"{self.base_url}/api/embed",
                json={"model": self.embed_model, "input": texts},
            )
        if resp.status_code != 200:
            raise OllamaError(f"embed failed: {resp.status_code} {resp.text[:200]}")
        data = resp.json()
        vectors = data.get("embeddings")
        if not vectors or len(vectors) != len(texts):
            raise OllamaError("embed returned mismatched number of vectors")
        return [list(map(float, v)) for v in vectors]

    async def chat(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        options: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> str:
        sampling: dict[str, Any] = {
            "temperature": temperature,
            "num_ctx": self.num_ctx,
            "top_p": settings.llm_top_p,
            "repeat_penalty": settings.llm_repeat_penalty,
        }
        if seed is None:
            seed = settings.llm_seed
        sampling["seed"] = seed
        if options:
            sampling.update(options)
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            "options": sampling,
        }
        async with self._semaphore, httpx.AsyncClient(timeout=self.chat_timeout) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
        if resp.status_code != 200:
            raise OllamaError(f"chat failed: {resp.status_code} {resp.text[:200]}")
        return resp.json()["message"]["content"].strip()

    async def ensure_model_ready(self) -> None:
        """Fail-fast: модель chat_model должна быть доступна И загружаема.

        /api/show проверяет только наличие манифеста (без загрузки весов) —
        поэтому дополнительно прогреваем модель крошечным чатом с keep_alive,
        чтобы нехватка RAM/OOM проявилась на старте, а не на первом запросе.
        """
        async with httpx.AsyncClient(timeout=self.chat_timeout) as client:
            try:
                resp = await client.post(
                    f"{self.base_url}/api/show", json={"name": self.chat_model}
                )
            except httpx.HTTPError as exc:
                raise OllamaError(
                    f"Ollama недоступен на {self.base_url}: {exc}. "
                    "Проверь: open -a OrbStack; ollama serve."
                ) from exc
        if resp.status_code != 200:
            raise OllamaError(
                f"Модель '{self.chat_model}' недоступна в Ollama "
                f"(HTTP {resp.status_code}). Выполни: ollama pull {self.chat_model}"
            )
        # Прогрев: реальная загрузка весов. OOM случится здесь, не у пользователя.
        warm = await self.chat(
            [{"role": "user", "content": "ок"}],
            temperature=0.0,
            options={"num_predict": 1, "keep_alive": "30m"},
        )
        if warm is None:
            raise OllamaError(f"Модель '{self.chat_model}' вернула пустой ответ при прогреве")


ollama = OllamaClient()
