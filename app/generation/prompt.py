from __future__ import annotations

from app.schemas import ChunkResult

SYSTEM_PROMPT = (
    "Ты — корпоративный помощник. Отвечай ТОЛЬКО на основе предоставленного контекста.\n"
    "Правила:\n"
    "1. Если в контексте достаточно информации — дай краткий, человечный ответ.\n"
    "2. Каждое утверждение сопровождай ссылкой-источником в формате [c<id>], "
    "где <id> — идентификатор чанка из контекста (например [c42]).\n"
    "3. Если в контексте НЕТ нужной информации — ответь ровно одним маркером: "
    "НЕВОЗМОЖНО_ОТВЕТИТЬ (без пояснений).\n"
    "4. Запрещено придумывать факты, цифры и нормы, которых нет в контексте.\n"
    "5. Запрещено использовать id чанков, которых нет в контексте."
)

CANNOT_ANSWER = "НЕВОЗМОЖНО_ОТВЕТИТЬ"


def build_context(chunks: list[ChunkResult]) -> str:
    lines = []
    for c in chunks:
        meta = f"стр. {c.page}" if c.page is not None else "без стр."
        header = f"[c{c.chunk_id}] ({meta}, Документ: {c.source_name})"
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
