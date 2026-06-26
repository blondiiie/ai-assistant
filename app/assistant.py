from __future__ import annotations

import contextlib
from dataclasses import dataclass

from app.config import settings
from app.generation.service import STUB_ANSWER
from app.generation.service import answer as generate_answer
from app.retrieval.service import search as retrieve
from app.schemas import ChunkResult
from app.sync.scanner import ensure_fresh

_BROAD_HINTS = ("всё", "всех", "все", "подробнее", "расскажи", "опиши", "целиком", "полностью")


def _is_broad(question: str) -> bool:
    lowered = question.lower()
    return any(hint in lowered for hint in _BROAD_HINTS)


@dataclass
class AskOutcome:
    answer: str
    grounded: bool
    sources: list[ChunkResult]


async def ask(question: str) -> AskOutcome:
    if settings.source_list:
        with contextlib.suppress(Exception):
            await ensure_fresh()
    top_k = settings.top_k_broad if _is_broad(question) else None
    retrieved = await retrieve(question, top_k=top_k)
    if not retrieved.found:
        return AskOutcome(answer=STUB_ANSWER, grounded=False, sources=[])

    generated = await generate_answer(question, retrieved.results)
    if not generated.grounded:
        return AskOutcome(answer=STUB_ANSWER, grounded=False, sources=[])

    by_id = {c.chunk_id: c for c in retrieved.results}
    sources = [by_id[cid] for cid in generated.cited_chunk_ids if cid in by_id]
    return AskOutcome(answer=generated.answer, grounded=True, sources=sources)
