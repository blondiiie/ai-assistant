from __future__ import annotations

import logging
import re

from app.config import settings
from app.generation.grounding import (
    CITE_TAG_RE,
    _content_tokens,
    filter_answer,
    grounding_components,
    grounding_score,
    has_foreign_script,
    is_refusal,
    normalize_inline_lists,
    split_sentences,
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

# Этап 2.2: развёрнутый ответ, к которому применима посентенсная фильтрация
# с min_kept_sentences. Короткие (да/нет/термин/число) проходят единый скоринг.
_LONG_ANSWER_MIN_SENTENCES = 3


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
    # Этап 6.2: нормализуем инлайн-списки ДО скоринга/фильтрации — чтобы
    # grounding видел корректную структуру (отдельные пункты), а не слитный
    # текст. Модель qwen2.5:3b часто выдаёт перечисления одной строкой.
    cleaned = normalize_inline_lists(cleaned)
    return cleaned


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


def _log_grounding(label: str, cleaned: str, all_contents: list[str], score: float) -> None:
    """Этап 3.1: структурированный лог решения grounding для отладки."""
    comps = grounding_components(cleaned, all_contents)
    logger.info(
        "grounding[%s score=%.3f threshold=%.2f "
        "word_coverage=%.2f trigram_overlap=%.2f distinctive_tokens=%.2f len=%d sents=%d]",
        label,
        score,
        settings.grounding_threshold,
        comps["word_coverage"],
        comps["trigram_overlap"],
        comps["distinctive_tokens"],
        len(cleaned),
        len(split_sentences(cleaned)),
    )


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
    top_p: float | None,
    seed: int | None,
    label: str,
) -> tuple[GenerateResult | None, bool, float]:
    """Возвращает (результат|None, был_ли_отказ, последний_score).

    Этап 2.1: единый grounding_score вместо AND-каскада.
    Этап 2.2: короткие ответы (< 3 предложений) проходят без посентенсной
    фильтрации и min_kept_sentences.
    Этап 2.3: параметры сэмплинга (temp/top_p/seed) управляются вызывающей
    стороной — ретраи диверсифицируются, попытка 0 детерминирована.
    """
    raw = await ollama.chat(messages, temperature=temperature, top_p=top_p, seed=seed)
    logger.info("generation[%s]: raw_len=%d temp=%.2f", label, len(raw), temperature)
    if is_refusal(raw):
        logger.info("generation[%s]: refusal", label)
        return None, True, 0.0
    cleaned = _clean(raw)
    if not cleaned:
        return None, False, 0.0
    if is_refusal(cleaned):
        return None, True, 0.0

    # Этап 2.1: жёсткий сейф — посторонняя письменность отсекает попытку.
    if has_foreign_script(cleaned):
        logger.warning("grounding[%s]: посторонняя письменность -> отсев", label)
        return None, False, 0.0

    score = grounding_score(cleaned, all_contents)
    _log_grounding(label, cleaned, all_contents, score)

    if score < settings.grounding_threshold:
        logger.info(
            "grounding[%s]: провал единого порога (%.3f < %.2f)",
            label,
            score,
            settings.grounding_threshold,
        )
        return None, False, score

    # Этап 2.2: короткие ответы — без посентенсной фильтрации и min_kept_sentences.
    is_short = len(split_sentences(cleaned)) < _LONG_ANSWER_MIN_SENTENCES
    if is_short:
        refined = cleaned
    else:
        refined = _refine(cleaned, all_contents)
        if refined is None:
            logger.info("grounding[%s]: insufficient после посентенсной фильтрации", label)
            return None, False, score

    src_ids = _source_ids(refined, context_chunks, settings.source_overlap)
    return GenerateResult(answer=refined, cited_chunk_ids=src_ids, grounded=True), False, score


async def answer(question: str, context_chunks: list[ChunkResult]) -> GenerateResult:
    if not context_chunks:
        return GenerateResult(answer=STUB_ANSWER, cited_chunk_ids=[], grounded=False)

    messages = build_messages(question, context_chunks)
    recovery_messages = build_recovery_messages(question, context_chunks)
    all_contents = [c.content for c in context_chunks]

    # --- Этап 2.3: осмысленные ретраи (сломать детерминизм ретраев) ---
    # Попытка 0: temp=0, seed=llm_seed, top_p=llm_top_p (воспроизводимость).
    # Попытки 1..N: рост temp + seed+attempt + top_p=llm_retry_top_p.
    refused = False
    best_score = 0.0
    result, was_refusal, best_score = await _try_generate(
        messages, all_contents, context_chunks,
        temperature=0.0, top_p=settings.llm_top_p, seed=settings.llm_seed,
        label="main#0",
    )
    refused = was_refusal
    if result is not None:
        return result

    for attempt in range(1, settings.grounding_max_retries + 1):
        retry_temp = min(0.4, settings.llm_retry_base_temperature * attempt)
        retry_seed = settings.llm_seed + attempt
        result, was_refusal, score = await _try_generate(
            messages, all_contents, context_chunks,
            temperature=retry_temp, top_p=settings.llm_retry_top_p, seed=retry_seed,
            label=f"main#{attempt}",
        )
        refused = refused or was_refusal
        best_score = max(best_score, score)
        if result is not None:
            return result

    # --- Recovery (Этап 2.5): при отказе ИЛИ при близком скоринге ---
    # Близкий скоринг: ответ почти опирается на контекст, но чуть не дотянул.
    recovery_band = settings.grounding_threshold - settings.grounding_recovery_margin
    if refused or best_score >= recovery_band:
        recovery, _, _ = await _try_generate(
            recovery_messages, all_contents, context_chunks,
            temperature=settings.llm_recovery_temperature,
            top_p=settings.llm_top_p, seed=settings.llm_seed,
            label="recovery",
        )
        if recovery is not None:
            return recovery

    # --- Yes/no fallback: модель ложно отказывается на выводных да/нет-вопросах ---
    if _is_yesno(question):
        yesno, _, _ = await _try_generate(
            build_yesno_messages(question, context_chunks),
            all_contents,
            context_chunks,
            temperature=settings.llm_yesno_temperature,
            top_p=settings.llm_top_p, seed=settings.llm_seed,
            label="yesno",
        )
        if yesno is not None:
            return yesno

    return GenerateResult(answer=STUB_ANSWER, cited_chunk_ids=[], grounded=False)