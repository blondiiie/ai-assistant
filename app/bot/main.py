from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message
from sqlalchemy import func, select

from app.assistant import ask
from app.config import settings
from app.db.models import Chunk, Document
from app.db.session import async_session
from app.ingest.parsers import SUPPORTED_EXTENSIONS
from app.ingest.service import store
from app.sync.scanner import scan as scan_sources

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

INLINE_CITE_RE = re.compile(r"[ \t]*\[c\d+\][ \t]*")
CONTEXT_HEADER_RE = re.compile(r"[ \t]*\([^)]*Файл:\s*[^)]*\)[ \t]*:?[ \t]*")
_SOURCE_EXT = (".md", ".pdf", ".docx", ".txt")


def _display_source(name: str) -> str:
    for ext in _SOURCE_EXT:
        if name.endswith(ext):
            return name[: -len(ext)]
    return name


def _format(outcome) -> str:
    text = CONTEXT_HEADER_RE.sub(" ", outcome.answer)
    text = INLINE_CITE_RE.sub(" ", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if not outcome.grounded or not outcome.sources:
        return text
    seen: set[tuple[str, str]] = set()
    footer: list[str] = []
    for s in outcome.sources:
        if s.section:
            loc = f"раздел «{s.section}»"
        elif s.page:
            loc = f"стр. {s.page}"
        else:
            loc = "без раздела"
        source = _display_source(s.source_name)
        key = (source, loc)
        if key in seen:
            continue
        seen.add(key)
        footer.append(f"[Источник: {source} · {loc}]")
    return f"{text}\n\n" + "\n".join(footer)


def _is_allowed(message: Message) -> bool:
    if not settings.allowed_chats:
        return settings.open_access
    return message.chat.id in settings.allowed_chats


async def cmd_start(message: Message) -> None:
    await message.answer(
        "Привет! Я корпоративный ассистент. Задай вопрос по регламентам — "
        "отвечу только на основе загруженных документов."
    )


async def cmd_help(message: Message) -> None:
    await message.answer(
        "Просто напиши свой вопрос текстом. Я найду ответ в регламентах "
        "и укажу источник. Если информации нет — честно скажу.\n\n"
        "Админам: /upload — как загрузить документ."
    )


async def cmd_upload(message: Message) -> None:
    if not _is_allowed(message):
        await message.answer("Доступ запрещён. Обратитесь к администратору.")
        return
    await message.answer(
        "Пришли файл документа (PDF, DOCX или TXT) сообщением. "
        "Я его проиндексирую — после этого по нему можно будет задавать вопросы.\n\n"
        "Также: /rescan — переиндексировать источники, /status — статистика."
    )


async def cmd_rescan(message: Message) -> None:
    if not _is_allowed(message):
        await message.answer("Доступ запрещён. Обратитесь к администратору.")
        return
    await message.answer("Сканирую источники…")
    try:
        result = await scan_sources()
    except Exception:
        logger.exception("rescan failed for chat %s", message.chat.id)
        await message.answer("Ошибка при сканировании. Подробности в логах.")
        return
    text = f"Готово. {result.summary()}"
    if result.errors:
        text += "\n\nОшибки:\n" + "\n".join(result.errors[:5])
    await message.answer(text)


async def cmd_status(message: Message) -> None:
    if not _is_allowed(message):
        await message.answer("Доступ запрещён. Обратитесь к администратору.")
        return
    async with async_session() as session:
        docs = (
            await session.execute(
                select(func.count()).select_from(Document).where(Document.active.is_(True))
            )
        ).scalar_one()
        chunks = (
            await session.execute(
                select(func.count(Chunk.id)).join(Document).where(Document.active.is_(True))
            )
        ).scalar_one()
    sources = ", ".join(settings.source_list) or "(не заданы)"
    await message.answer(
        f"Источники: {sources}\nАктивных документов: {docs}\nАктивных чанков: {chunks}"
    )


async def handle_document(message: Message) -> None:
    if not _is_allowed(message):
        await message.answer("Доступ запрещён. Обратитесь к администратору.")
        return
    doc = message.document
    if doc is None:
        return

    filename = doc.file_name or "document"
    ext = Path(filename).suffix.lower().lstrip(".")
    doc_type = SUPPORTED_EXTENSIONS.get(ext)
    if doc_type is None:
        await message.answer(
            f"Формат .{ext} не поддерживается. Доступно: PDF, DOCX, TXT."
        )
        return

    if doc.file_size and doc.file_size > settings.max_upload_mb * 1024 * 1024:
        await message.answer(f"Файл больше {settings.max_upload_mb} МБ. Уменьшите размер.")
        return

    await message.answer(f"Индексирую «{filename}»…")
    dest: Path | None = None
    try:
        tg_file = await message.bot.get_file(doc.file_id)
        buffer = await message.bot.download_file(tg_file.file_path)
        content = buffer.read()
        uploads = Path(settings.uploads_dir)
        uploads.mkdir(parents=True, exist_ok=True)
        dest = uploads / filename
        dest.write_bytes(content)
        document_id, chunks_created = await store(filename, doc_type, str(dest))
    except ValueError as exc:
        if dest is not None:
            dest.unlink(missing_ok=True)
        await message.answer(f"Не удалось обработать файл: {exc}")
        return
    except Exception:
        logger.exception("upload failed for chat %s", message.chat.id)
        await message.answer("Ошибка при индексации файла. Подробности в логах.")
        return

    await message.answer(
        f"✅ Готово. Документ «{filename}» добавлен (id={document_id}), "
        f"чанков: {chunks_created}. Теперь можно задавать по нему вопросы."
    )


async def handle_question(message: Message) -> None:
    if not _is_allowed(message):
        await message.answer("Доступ запрещён. Обратитесь к администратору.")
        return
    question = (message.text or "").strip()
    if not question:
        return
    await message.answer("Ищу по заметкам…")
    try:
        outcome = await ask(question)
    except Exception:
        logger.exception("ask() failed for chat %s", message.chat.id)
        await message.answer("Произошла ошибка при обработке запроса. Попробуйте позже.")
        return
    await message.answer(_format(outcome))


def build_dispatcher() -> Dispatcher:
    dp = Dispatcher()
    dp.message.register(cmd_start, Command("start"))
    dp.message.register(cmd_help, Command("help"))
    dp.message.register(cmd_upload, Command("upload"))
    dp.message.register(cmd_rescan, Command("rescan"))
    dp.message.register(cmd_status, Command("status"))
    dp.message.register(handle_document, F.document)
    dp.message.register(handle_question, F.text)
    return dp


async def main() -> None:
    if not settings.telegram_bot_token:
        print("TELEGRAM_BOT_TOKEN не задан в .env. Получите токен у @BotFather.")
        sys.exit(1)
    bot = Bot(settings.telegram_bot_token)
    dp = build_dispatcher()
    if settings.source_list:
        print("Стартовый скан источников…")
        try:
            result = await scan_sources()
            print("Скан: " + result.summary())
        except Exception:
            print("Стартовый скан завершился ошибкой — бот всё равно запускается.")
    else:
        print("SOURCE_DIRS не заданы — стартовый скан пропущен.")
    print("Бот запущен. Ctrl+C для остановки.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
