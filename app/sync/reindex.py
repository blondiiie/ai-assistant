from __future__ import annotations

import asyncio

from sqlalchemy import text

from app.db.session import async_session
from app.llm.tokens import OllamaTokenCounter, configure_counter
from app.sync.scanner import scan


async def reindex() -> None:
    """Полное переиндексирование: очистка документов/чанков/ссылок и новый скан.

    Нужно после изменений парсера/чанкования (напр. появление графа wikilinks),
    потому что обычный скан пропускает неизменные файлы по хэшу/mtime.
    """
    # Чанкование — по реальному токенайзеру модели (не cl100k).
    configure_counter(OllamaTokenCounter())
    async with async_session() as session, session.begin():
        await session.execute(text("DELETE FROM document_links"))
        await session.execute(text("DELETE FROM chunks"))
        await session.execute(text("DELETE FROM documents"))
    print("Хранилище очищено. Запускаю скан…")
    result = await scan()
    print("Готово. " + result.summary())
    if result.errors:
        print("Ошибки:\n" + "\n".join(result.errors))


if __name__ == "__main__":
    asyncio.run(reindex())
