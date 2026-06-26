from __future__ import annotations

import re

from app.config import settings
from app.generation.grounding import check, is_supported
from app.generation.prompt import CANNOT_ANSWER, build_messages
from app.llm.client import ollama
from app.schemas import ChunkResult, GenerateResult

STUB_ANSWER = (
    "В моих заметках нет информации по этому вопросу. "
    "Возможно, такой заметки пока нет — попробуй переформулировать или добавь заметку."
)

CITE_TAG_RE = re.compile(r"\[c\d+\]")


def _strip_invalid(raw: str, invalid_ids: set[int]) -> str:
    def repl(m: re.Match[str]) -> str:
        return "" if int(m.group()[2:-1]) in invalid_ids else m.group()

    return CITE_TAG_RE.sub(repl, raw)


def _clean(raw: str) -> str:
    cleaned = raw.replace(CANNOT_ANSWER, "").strip()
    return cleaned or STUB_ANSWER


async def answer(question: str, context_chunks: list[ChunkResult]) -> GenerateResult:
    if not context_chunks:
        return GenerateResult(answer=STUB_ANSWER, cited_chunk_ids=[], grounded=False)

    context_ids = {c.chunk_id for c in context_chunks}
    messages = build_messages(question, context_chunks)

    for _ in range(settings.grounding_max_retries + 1):
        raw = await ollama.chat(messages, temperature=0.0)
        if CANNOT_ANSWER in raw:
            return GenerateResult(answer=STUB_ANSWER, cited_chunk_ids=[], grounded=False)
        valid, has_any, invalid = check(raw, context_ids)
        if has_any:
            cleaned = _strip_invalid(raw, invalid)
            cited_contents = [c.content for c in context_chunks if c.chunk_id in set(valid)]
            if is_supported(cleaned, cited_contents, settings.grounding_min_overlap):
                return GenerateResult(
                    answer=_clean(cleaned), cited_chunk_ids=valid, grounded=True
                )

    return GenerateResult(answer=STUB_ANSWER, cited_chunk_ids=[], grounded=False)
