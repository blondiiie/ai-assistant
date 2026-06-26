from __future__ import annotations

from app.schemas import ChunkResult

SYSTEM_PROMPT = (
    "Ты — личный помощник для поиска информации по заметкам и файлам пользователя. "
    "Отвечай ТОЛЬКО на основе предоставленного контекста.\n"
    "Правила:\n"
    "1. Если в контексте достаточно информации — дай ясный, точный ответ по сути.\n"
    "2. Внутри ответа помечай каждое утверждение тегом источника в формате [c<id>], "
    "где <id> — идентификатор чанка из контекста (например [c42]). Это нужно для проверки.\n"
    "3. Если в контексте НЕТ нужной информации — ответь ровно одним маркером: "
    "НЕВОЗМОЖНО_ОТВЕТИТЬ (без пояснений).\n"
    "4. Запрещено придумывать факты, которых нет в контексте.\n"
    "5. Запрещено использовать id чанков, которых нет в контексте."
)

CANNOT_ANSWER = "НЕВОЗМОЖНО_ОТВЕТИТЬ"


def build_context(chunks: list[ChunkResult]) -> str:
    lines = []
    for c in chunks:
        meta = []
        if c.page is not None:
            meta.append(f"стр. {c.page}")
        if c.section:
            meta.append(f"раздел «{c.section}»")
        meta_str = ", ".join(meta) if meta else "без метаданных"
        header = f"[c{c.chunk_id}] ({meta_str}, Файл: {c.source_name})"
        lines.append(f"{header}: {c.content}")
    return "\n\n".join(lines)


def build_messages(question: str, chunks: list[ChunkResult]) -> list[dict[str, str]]:
    context = build_context(chunks)
    user = (
        f"Контекст:\n{context}\n\n"
        f"Вопрос пользователя: {question}\n\n"
        f"Ответь по правилам системы."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]
