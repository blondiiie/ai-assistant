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
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.chat_model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx, **(options or {})},
        }
        async with self._semaphore, httpx.AsyncClient(timeout=self.chat_timeout) as client:
            resp = await client.post(f"{self.base_url}/api/chat", json=payload)
        if resp.status_code != 200:
            raise OllamaError(f"chat failed: {resp.status_code} {resp.text[:200]}")
        return resp.json()["message"]["content"].strip()


ollama = OllamaClient()
