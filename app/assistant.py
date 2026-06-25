from __future__ import annotations

from dataclasses import dataclass

from app.generation.service import STUB_ANSWER
from app.generation.service import answer as generate_answer
from app.retrieval.service import search as retrieve
from app.schemas import ChunkResult


@dataclass
class AskOutcome:
    answer: str
    grounded: bool
    sources: list[ChunkResult]


async def ask(question: str) -> AskOutcome:
    retrieved = await retrieve(question)
    if not retrieved.found:
        return AskOutcome(answer=STUB_ANSWER, grounded=False, sources=[])

    generated = await generate_answer(question, retrieved.results)
    if not generated.grounded:
        return AskOutcome(answer=STUB_ANSWER, grounded=False, sources=[])

    by_id = {c.chunk_id: c for c in retrieved.results}
    sources = [by_id[cid] for cid in generated.cited_chunk_ids if cid in by_id]
    return AskOutcome(answer=generated.answer, grounded=True, sources=sources)
