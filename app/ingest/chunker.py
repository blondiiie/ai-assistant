from __future__ import annotations

from app.config import settings
from app.ingest.parsers import TextBlock
from app.llm.tokens import count_tokens, get_counter
from app.schemas import ChunkMeta


def _split_pieces(pieces: list[str], size: int, overlap: int) -> list[str]:
    """Точная нарезка по кусочкам-токенам модели (один запрос на блок текста).

    Окна восстанавливаются склейкой кусков-токенов — decode не требуется.
    """
    if not pieces:
        return []
    step = max(1, size - overlap)
    out: list[str] = []
    for start in range(0, len(pieces), step):
        window = pieces[start : start + size]
        if not window:
            break
        out.append("".join(window).strip())
        if start + size >= len(pieces):
            break
    return [p for p in out if p.strip()]


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    """Нарезает текст окном по токенам модели; размер — в токенах.

    Использует реальный токенайзер (через get_counter().tokenize): cl100k_base
    намеренно НЕ применяется — он занижает кириллицу/Qwen, что приводило к
    переполнению контекстного окна. Офлайн (без Ollama) или если Ollama вернул
    int-id — переключается на разбиение по словам с оценкой count_tokens.
    """
    if not text:
        return []
    counter = get_counter()
    pieces = counter.tokenize(text)
    if pieces:
        exact = _split_pieces(pieces, size, overlap)
        if exact:
            return exact
    return _word_window(text, size, overlap)


def _word_window(text: str, size: int, overlap: int) -> list[str]:
    """Резервная нарезка по словам; размер оценивается через count_tokens."""
    words = text.split()
    if not words:
        return []
    step = max(1, size - overlap)
    pieces: list[str] = []
    buf: list[str] = []
    n = 0
    for w in words:
        tn = count_tokens(w + " ") or 1
        if n + tn > size and buf:
            pieces.append(" ".join(buf))
            while buf and n > step:
                removed = count_tokens(buf[0] + " ") or 1
                buf.pop(0)
                n -= removed
        buf.append(w)
        n += tn
    if buf:
        pieces.append(" ".join(buf))
    return [p for p in pieces if p.strip()]


def chunk_blocks(
    blocks: list[TextBlock],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[ChunkMeta]:
    """Синхронное чанкование (офлайн-эвристика / прогретый кеш).

    Для продакшен-токенайзера (Ollama) используйте async-вариант
    `achunk_blocks`, который токенизирует блоки без блокировки event loop.
    """
    metas: list[ChunkMeta] = []
    idx = 0
    for block in blocks:
        for piece in _split_text(block.text, chunk_size, chunk_overlap):
            piece = piece.strip()
            if not piece:
                continue
            metas.append(
                ChunkMeta(chunk_index=idx, content=piece, page=block.page, section=block.section)
            )
            idx += 1
    return metas


async def achunk_blocks(
    blocks: list[TextBlock],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[ChunkMeta]:
    """Async-чанкование: токенизирует каждый блок реальным токенайзером модели
    без блокировки event loop. Используется в ingest-pipeline."""
    from app.llm.tokens import OllamaTokenCounter  # noqa: PLC0415

    metas: list[ChunkMeta] = []
    idx = 0
    counter = get_counter()
    for block in blocks:
        pieces = None
        if isinstance(counter, OllamaTokenCounter):
            pieces = await counter.atokenize(block.text)
        if pieces:
            chunks = _split_pieces(pieces, chunk_size, chunk_overlap)
        else:
            chunks = _word_window(block.text, chunk_size, chunk_overlap)
        for piece in chunks:
            piece = piece.strip()
            if not piece:
                continue
            metas.append(
                ChunkMeta(chunk_index=idx, content=piece, page=block.page, section=block.section)
            )
            idx += 1
    return metas


# Используется retrieval для оценки «хвоста» при скольжении окна (offline-safe).
__all__ = ["_split_text", "chunk_blocks", "achunk_blocks"]
