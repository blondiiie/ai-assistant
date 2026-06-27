from __future__ import annotations

import logging
import re

from app.config import settings
from app.generation.grounding import (
    CITE_TAG_RE,
    _content_tokens,
    filter_answer,
    has_foreign_script,
    is_refusal,
    is_supported,
    is_word_supported,
    missing_distinctive_tokens,
)
from app.generation.prompt import (
    CANNOT_ANSWER,
    build_messages,
    build_recovery_messages,
    build_yesno_messages,
)
from app.llm.client import ollama
from app.schemas import ChunkResult, GenerateResult

logger = logging.getLogger(__name__)

STUB_ANSWER = (
    "В моих заметках нет информации по этому вопросу. "
    "Возможно, такой заметки пока нет — попробуй переформулировать или добавь заметку."
)

_YESNO_RE = re.compile(
    r"\b(?:это|является\s+ли|обязательно\s+ли|надо\s+ли|нужно\s+ли|"
    r"должно\s+ли|правда\s+ли|верно\s+ли)\b",
    re.IGNORECASE,
)
_WH_STEMS = {
    "какие", "какой", "какая", "какое", "чем", "что", "сколько", "кто",
    "где", "когда", "почему", "зачем", "перечисли", "перечислите", "список",
    "расскажи", "опиши",
}


def _is_yesno(question: str) -> bool:
    q = question.strip().lower()
    if not q.endswith("?"):
        return False
    # Не да/нет, если есть wh-слово или триггер перечисления/обзора.
    words = set(re.findall(r"[а-яё]+", q))
    if words & _WH_STEMS:
        return False
    return bool(_YESNO_RE.search(q)) or q.startswith(("это ", "это—", "это-"))


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


def _passes_gates(cleaned: str, all_contents: list[str]) -> bool:
    if has_foreign_script(cleaned):
        logger.warning("grounding: посторонняя письменность в ответе -> отсев попытки")
        return False
    if not is_supported(cleaned, all_contents, settings.grounding_min_overlap):
        return False
    if not is_word_supported(cleaned, all_contents, settings.grounding_word_coverage):
        return False
    return not missing_distinctive_tokens(cleaned, all_contents)


def _refine(cleaned: str, all_contents: list[str]) -> str | None:
    """Посентенсная фильтрация. None = недостаточно смысла."""
    return filter_answer(
        cleaned,
        all_contents,
        min_coverage=settings.grounding_sentence_coverage,
        min_kept_ratio=settings.grounding_min_kept_ratio,
        min_kept_sentences=settings.grounding_min_kept_sentences,
    )


async def _try_generate(
    messages,
    all_contents: list[str],
    context_chunks: list[ChunkResult],
    *,
    temperature: float,
    label: str,
) -> tuple[GenerateResult | None, bool]:
    """Возвращает (результат, был_ли_отказ). Результат None — повторить."""
    raw = await ollama.chat(messages, temperature=temperature)
    logger.info("generation[%s]: raw_len=%d", label, len(raw))
    if is_refusal(raw):
        logger.info("generation[%s]: refusal", label)
        return None, True
    cleaned = _clean(raw)
    if not cleaned or cleaned == STUB_ANSWER or is_refusal(cleaned):
        return None, False
    if not _passes_gates(cleaned, all_contents):
        logger.info("generation[%s]: провалены целевые grounding-гейты", label)
        return None, False
    refined = _refine(cleaned, all_contents)
    if refined is None:
        logger.info("generation[%s]: insufficient после посентенсной фильтрации", label)
        return None, False
    src_ids = _source_ids(refined, context_chunks, settings.source_overlap)
    return GenerateResult(answer=refined, cited_chunk_ids=src_ids, grounded=True), False


async def answer(question: str, context_chunks: list[ChunkResult]) -> GenerateResult:
    if not context_chunks:
        return GenerateResult(answer=STUB_ANSWER, cited_chunk_ids=[], grounded=False)

    messages = build_messages(question, context_chunks)
    recovery_messages = build_recovery_messages(question, context_chunks)
    all_contents = [c.content for c in context_chunks]

    # Все grounded-попытки на temp=0 + seed: воспроизводимость между запусками.
    # Повышение температуры для «лечения» grounding концептуально вредно.
    refused = False
    for attempt in range(settings.grounding_max_retries + 1):
        result, was_refusal = await _try_generate(
            messages, all_contents, context_chunks,
            temperature=0.0, label=f"main#{attempt}",
        )
        refused = refused or was_refusal
        if result is not None:
            return result

    # Recovery: ТОЛЬКО против ложных отказов (NOANSWER), низкая температура.
    if refused:
        recovery, _ = await _try_generate(
            recovery_messages, all_contents, context_chunks,
            temperature=settings.llm_recovery_temperature, label="recovery",
        )
        if recovery is not None:
            return recovery

    # Yes/no fallback: модель 7b ложно отказывается на выводных да/нет-вопросах.
    if _is_yesno(question):
        yesno, _ = await _try_generate(
            build_yesno_messages(question, context_chunks),
            all_contents,
            context_chunks,
            temperature=settings.llm_yesno_temperature,
            label="yesno",
        )
        if yesno is not None:
            return yesno

    return GenerateResult(answer=STUB_ANSWER, cited_chunk_ids=[], grounded=False)
