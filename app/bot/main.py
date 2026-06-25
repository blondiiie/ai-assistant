from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message

from app.assistant import ask
from app.config import settings
from app.ingest.parsers import SUPPORTED_EXTENSIONS
from app.ingest.service import store

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

INLINE_CITE_RE = re.compile(r"\s*\[c\d+\]\s*")


def _format(outcome) -> str:
    text = INLINE_CITE_RE.sub(" ", outcome.answer).strip()
    text = re.sub(r"\s{2,}", " ", text)
    if not outcome.grounded:
        return text
    seen: set[tuple[str, int | None]] = set()
    footer: list[str] = []
    for s in outcome.sources:
        key = (s.source_name, s.page)
        if key in seen:
            continue
        seen.add(key)
        loc = f"стр. {s.page}" if s.page is not None else "без страницы"
        footer.append(f"📄 {s.source_name} ({loc})")
    if footer:
        return f"{text}\n\nИсточники:\n" + "\n".join(footer)
    return text


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
        "Я его проиндексирую — после этого по нему можно будет задавать вопросы."
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
    await message.answer("Ищу в регламентах…")
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
    dp.message.register(handle_document, F.document)
    dp.message.register(handle_question, F.text)
    return dp


async def main() -> None:
    if not settings.telegram_bot_token:
        print("TELEGRAM_BOT_TOKEN не задан в .env. Получите токен у @BotFather.")
        sys.exit(1)
    bot = Bot(settings.telegram_bot_token)
    dp = build_dispatcher()
    print("Бот запущен. Ctrl+C для остановки.")
    await dp.start_polling(bot)


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
