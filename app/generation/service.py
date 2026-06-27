from __future__ import annotations

import re

from app.config import settings
from app.generation.grounding import (
    _content_tokens,
    is_refusal,
    is_supported,
    is_word_supported,
    missing_distinctive_tokens,
)
from app.generation.prompt import CANNOT_ANSWER, build_messages, build_recovery_messages
from app.llm.client import ollama
from app.schemas import ChunkResult, GenerateResult

STUB_ANSWER = (
    "В моих заметках нет информации по этому вопросу. "
    "Возможно, такой заметки пока нет — попробуй переформулировать или добавь заметку."
)

CITE_TAG_RE = re.compile(r"\[c\d+\]")


def _clean(raw: str) -> str:
    cleaned = CITE_TAG_RE.sub("", raw).replace(CANNOT_ANSWER, "").strip()
    return cleaned or STUB_ANSWER


def _source_ids(answer: str, context_chunks: list[ChunkResult], min_overlap: float) -> list[int]:
    answer_tokens = set(_content_tokens(answer))
    if not answer_tokens:
        return [context_chunks[0].chunk_id]
    scored = []
    for c in context_chunks:
        chunk_tokens = set(_content_tokens(c.content))
        overlap = len(answer_tokens & chunk_tokens) / len(answer_tokens)
        if overlap >= min_overlap:
            scored.append((overlap, c.chunk_id))
    scored.sort(reverse=True)
    return [cid for _, cid in scored] or [context_chunks[0].chunk_id]


async def answer(question: str, context_chunks: list[ChunkResult]) -> GenerateResult:
    if not context_chunks:
        return GenerateResult(answer=STUB_ANSWER, cited_chunk_ids=[], grounded=False)

    messages = build_messages(question, context_chunks)
    recovery_messages = build_recovery_messages(question, context_chunks)
    all_contents = [c.content for c in context_chunks]

    refused = False
    for attempt in range(settings.grounding_max_retries + 1):
        temperature = 0.0 if attempt == 0 else 0.3
        raw = await ollama.chat(messages, temperature=temperature)
        if is_refusal(raw):
            refused = True
            continue
        cleaned = _clean(raw)
        if not cleaned or cleaned == STUB_ANSWER or is_refusal(cleaned):
            continue
        if not is_supported(cleaned, all_contents, settings.grounding_min_overlap):
            continue
        if not is_word_supported(cleaned, all_contents, settings.grounding_word_coverage):
            continue
        if missing_distinctive_tokens(cleaned, all_contents):
            continue
        src_ids = _source_ids(cleaned, context_chunks, settings.source_overlap)
        return GenerateResult(answer=cleaned, cited_chunk_ids=src_ids, grounded=True)

    # Защита от ложных отказов: модель отказалась, хотя контекст релевантен.
    # Дополнительная попытка с инструкцией «не отказывайся, выводи из контекста».
    if refused:
        raw = await ollama.chat(recovery_messages, temperature=0.3)
        if not is_refusal(raw):
            cleaned = _clean(raw)
            if (
                cleaned
                and cleaned != STUB_ANSWER
                and not is_refusal(cleaned)
                and is_supported(cleaned, all_contents, settings.grounding_min_overlap)
                and is_word_supported(cleaned, all_contents, settings.grounding_word_coverage)
                and not missing_distinctive_tokens(cleaned, all_contents)
            ):
                src_ids = _source_ids(cleaned, context_chunks, settings.source_overlap)
                return GenerateResult(answer=cleaned, cited_chunk_ids=src_ids, grounded=True)

    return GenerateResult(answer=STUB_ANSWER, cited_chunk_ids=[], grounded=False)
