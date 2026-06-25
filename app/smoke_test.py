"""Точка входа: smoke-тест подключения к Ollama (эмбеддинг + чат).

Запуск: uv run python -m app.smoke_test
Критерий приёмки Фазы 0: возвращается вектор и ответ от Ollama.
"""
from __future__ import annotations

import asyncio

import httpx

from app.config import settings


async def check_ollama() -> None:
    base = settings.ollama_url
    probe = "За сколько дней подавать заявление на отпуск?"

    async with httpx.AsyncClient(timeout=settings.embed_timeout) as client:
        # 1) Эмбеддинг
        emb_resp = await client.post(
            f"{base}/api/embed",
            json={"model": settings.embed_model, "input": [probe]},
        )
        emb_resp.raise_for_status()
        vector = emb_resp.json()["embeddings"][0]
        print(f"[OK] embeddings model={settings.embed_model} dim={len(vector)}")
        assert len(vector) == settings.embed_dimensions, (
            f"ожидалась размерность {settings.embed_dimensions}, "
            f"получено {len(vector)}. Проверьте embed_model."
        )

    async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
        # 2) Чат
        chat_resp = await client.post(
            f"{base}/api/chat",
            json={
                "model": settings.llm_model,
                "stream": False,
                "messages": [
                    {"role": "system", "content": "Ответь одним коротким предложением."},
                    {"role": "user", "content": probe},
                ],
            },
        )
        chat_resp.raise_for_status()
        answer = chat_resp.json()["message"]["content"]
        print(f"[OK] chat model={settings.llm_model}")
        print(f"      answer: {answer.strip()[:120]}")


async def main() -> None:
    print(f"Ollama URL: {settings.ollama_url}")
    print(f"LLM:    {settings.llm_model}")
    print(f"Embed:  {settings.embed_model} (dim={settings.embed_dimensions})")
    print("-" * 60)
    await check_ollama()
    print("-" * 60)
    print("Smoke-тест пройден.")


if __name__ == "__main__":
    asyncio.run(main())
