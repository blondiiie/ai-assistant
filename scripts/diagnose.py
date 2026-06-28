"""Диагностический CLI для RAG-пайплайна (Этап 3.2 рефакторинга).

Воспроизводит поведение бота без Telegram и показывает промежуточные этапы:
  1. Retrieval: top-k чанков с hybrid-score, sim, lex, токен-бюджет.
  2. Генерация: raw-ответ LLM.
  3. Grounding: единый score и его компоненты, решение гейта.
  4. Итог: grounded/stub + источник.

Запуск:
    uv run python -m scripts.diagnose "объясни мне REST"
    uv run python -m scripts.diagnose "расскажи всё про REST" --top-k 6

Требует поднятых Postgres и Ollama (open -a OrbStack; ollama serve).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from app.config import settings
from app.generation import service as gen_service
from app.generation.grounding import (
    grounding_components,
    has_foreign_script,
    is_refusal,
    split_sentences,
)
from app.generation.prompt import build_messages
from app.generation.service import _clean
from app.llm.client import ollama
from app.llm.tokens import acount_tokens
from app.retrieval.service import search

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("diagnose")


def _bar(value: float, width: int = 20) -> str:
    filled = int(round(value * width))
    return "█" * filled + "·" * (width - filled)


def _print_header(title: str) -> None:
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def _print_chunk(i: int, r) -> None:
    preview = r.content.replace("\n", " ")
    if len(preview) > 140:
        preview = preview[:137] + "..."
    print(
        f"  [{i}] id={r.chunk_id} score={r.score:.4f} "
        f"src={r.source_name} §={r.section}"
    )
    print(f"      {preview}")


async def diagnose(question: str, top_k: int | None) -> int:
    print(f"Вопрос: {question!r}")
    print(
        f"Модель: {settings.llm_model} | num_ctx={settings.llm_num_ctx} "
        f"reserve={settings.ctx_reserve} | top_k={top_k or settings.top_k}"
    )
    print(
        f"Grounding: threshold={settings.grounding_threshold} "
        f"word_coverage={settings.grounding_word_coverage} "
        f"sentence_coverage={settings.grounding_sentence_coverage}"
    )

    # --- 1. Retrieval ---
    _print_header("1. RETRIEVAL")
    retrieve = await search(question, top_k=top_k)
    if not retrieve.found:
        print("  НЕТ чанков выше sim_threshold — заглушка неизбежна.")
        return 1
    total_tokens = 0
    for i, r in enumerate(retrieve.results, 1):
        n = await acount_tokens(r.content)
        total_tokens += n
        _print_chunk(i, r)
        print(f"      ~{n} токенов")
    budget = settings.llm_num_ctx - settings.ctx_reserve
    print(f"\n  Итого чанков: {len(retrieve.results)} | ~{total_tokens} токенов | бюджет {budget}")

    # --- 2. Сырой ответ LLM ---
    _print_header("2. ГЕНЕРАЦИЯ (raw LLM, попытка 0: temp=0 seed=42)")
    messages = build_messages(question, retrieve.results)
    try:
        raw = await ollama.chat(
            messages, temperature=0.0, top_p=settings.llm_top_p, seed=settings.llm_seed
        )
    except Exception as exc:  # noqa: BLE001
        print(f"  Ollama ERROR: {exc}")
        return 2
    print(f"  raw_len={len(raw)}")
    print(f"  raw: {raw!r}")

    # --- 3. Grounding-анализ ---
    _print_header("3. GROUNDING-АНАЛИЗ (Этап 2.1 — единый score)")
    all_contents = [c.content for c in retrieve.results]
    if is_refusal(raw):
        print("  ОТКАЗ модели (is_refusal) → recovery/stub")
    cleaned = _clean(raw)
    if not cleaned:
        print("  ПУСТОЙ ответ после очистки → recovery/stub")
    else:
        foreign = has_foreign_script(cleaned)
        comps = grounding_components(cleaned, all_contents)
        score = sum(comps.values()) / len(comps)
        print(f"  has_foreign_script: {foreign}")
        wc = comps['word_coverage']
        tg = comps['trigram_overlap']
        dt = comps['distinctive_tokens']
        print(f"  word_coverage:     {wc:.3f} {_bar(wc)}")
        print(f"  trigram_overlap:   {tg:.3f} {_bar(tg)}")
        print(f"  distinctive_tok:   {dt:.3f} {_bar(dt)}")
        print("  ────────────────────────────────")
        print(f"  grounding_score:   {score:.3f} {_bar(score)}")
        verdict = "ПРИНЯТ" if score >= settings.grounding_threshold else "ПРОВАЛ"
        print(f"  порог: {settings.grounding_threshold:.2f} → {verdict}")
        print(f"  предложений: {len(split_sentences(cleaned))}")

    # --- 4. Итог (через полный сервисный путь с ретраями) ---
    _print_header("4. ИТОГ (полный пайплайн: ретраи + recovery)")
    result = await gen_service.answer(question, retrieve.results)
    print(f"  grounded: {result.grounded}")
    print(f"  cited_chunk_ids: {result.cited_chunk_ids}")
    print(f"  answer:\n{result.answer}")
    return 0 if result.grounded else 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Диагностика RAG-пайплайна")
    parser.add_argument("question", help="Вопрос для диагностики")
    parser.add_argument(
        "--top-k", type=int, default=None, help=f"Переопределить TOP_K (дефолт {settings.top_k})"
    )
    parser.add_argument(
        "--json", action="store_true", help="Вывести машиночитаемый JSON (кроме LLM-логов в stderr)"
    )
    args = parser.parse_args()

    try:
        code = asyncio.run(diagnose(args.question, args.top_k))
    except KeyboardInterrupt:
        code = 130
    if args.json:
        print(json.dumps({"exit_code": code}))
    sys.exit(code)


if __name__ == "__main__":
    main()