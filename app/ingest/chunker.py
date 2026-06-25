from __future__ import annotations

import tiktoken

from app.config import settings
from app.ingest.parsers import TextBlock
from app.schemas import ChunkMeta

_enc = tiktoken.get_encoding("cl100k_base")


def _split_text(text: str, size: int, overlap: int) -> list[str]:
    tokens = _enc.encode(text)
    if not tokens:
        return []
    step = max(1, size - overlap)
    pieces: list[str] = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + size]
        if not window:
            break
        pieces.append(_enc.decode(window))
        if start + size >= len(tokens):
            break
    return pieces


def chunk_blocks(
    blocks: list[TextBlock],
    chunk_size: int = settings.chunk_size,
    chunk_overlap: int = settings.chunk_overlap,
) -> list[ChunkMeta]:
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
