from __future__ import annotations

import asyncio
import json
import statistics
import time
from pathlib import Path

from app.assistant import ask
from app.config import settings
from app.retrieval.service import search

CORPUS = Path(__file__).resolve().parent.parent / "data" / "qa_corpus.json"
SWEEP = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def load_corpus() -> list[dict]:
    return json.loads(CORPUS.read_text(encoding="utf-8"))


async def sweep_thresholds(corpus: list[dict]) -> dict:
    rel = [c for c in corpus if c["type"] == "relevant"]
    oos = [c for c in corpus if c["type"] == "out_of_scope"]
    results = {}
    print("\n=== Калибровка sim_threshold (только retrieval, без LLM) ===")
    print(f"{'thresh':>7} | {'recall':>7} | {'oos_fp':>7}")
    for t in SWEEP:
        settings.sim_threshold = t
        rel_found = 0
        for c in rel:
            res = await search(c["q"])
            rel_found += int(res.found)
        oos_found = 0
        for c in oos:
            res = await search(c["q"])
            oos_found += int(res.found)
        recall = rel_found / len(rel)
        fp_rate = oos_found / len(oos)
        results[t] = (recall, fp_rate)
        print(f"{t:>7.2f} | {recall:>7.2%} | {fp_rate:>7.2%}")
    return results


def pick_threshold(results: dict) -> float:
    """Этап 4.1: выбрать sim_threshold с max recall при fp_rate ≤ 0.2.

    При равенстве recall/fp (частый случай на однородном корпусе) — выбираем
    МАКСИМАЛЬНЫЙ порог: он строже и в проде даёт меньше ложных срабатываний
    при том же recall. Ранее max() брал первый вставленный (минимальный) ключ —
    это был баг: выбиралось 0.20 даже когда 0.50 давал тот же recall.
    """
    candidates = [t for t, (recall, fp) in results.items() if recall >= 0.9 and fp <= 0.2]
    if candidates:
        return max(candidates)
    # Нет комбинации с fp≤0.2 — берём max recall, при равенстве min fp, потом max порог
    best_recall = max(r for r, _ in results.values())
    best = max(
        (t for t, (r, _) in results.items() if r == best_recall),
        key=lambda t: (-results[t][1], t),
    )
    return best


async def evaluate(corpus: list[dict], threshold: float) -> None:
    settings.sim_threshold = threshold
    print(f"\n=== End-to-end оценка при sim_threshold={threshold} ===")
    durations: list[float] = []
    anti_hallucination = 0
    rel_grounded = 0
    rel_keyword_hit = 0
    rel_with_source = 0
    n_rel = n_oos = 0
    for c in corpus:
        start = time.perf_counter()
        outcome = await ask(c["q"])
        durations.append(time.perf_counter() - start)
        if c["type"] == "out_of_scope":
            n_oos += 1
            anti_hallucination += int(not outcome.grounded)
        else:
            n_rel += 1
            if outcome.grounded:
                rel_grounded += 1
                rel_with_source += int(bool(outcome.sources))
                lowered = outcome.answer.lower()
                rel_keyword_hit += int(any(k.lower() in lowered for k in c["expect"]))
        flag = "OK " if (outcome.grounded == (c["type"] == "relevant")) else "XX "
        qtype = c["type"][:3]
        print(
            f"{flag}[{qtype}] {c['q'][:48]:<48} "
            f"grounded={outcome.grounded} t={durations[-1]:.1f}s"
        )

    p50 = statistics.median(durations)
    p95 = _percentile(durations, 95)
    print("\n--- Метрики ---")
    ah = anti_hallucination / n_oos
    print(f"Анти-галлюцинация (oos → заглушка): {anti_hallucination}/{n_oos} = {ah:.0%}")
    print("  цель ≥90%")
    print(f"Recall grounded (relevant → ответ): {rel_grounded}/{n_rel} = {rel_grounded/n_rel:.0%}")
    rg = rel_keyword_hit / n_rel
    print(f"Ответ содержит ключевой факт:       {rel_keyword_hit}/{n_rel} = {rg:.0%}")
    print(f"Источник приложен (для grounded):   {rel_with_source}/{rel_grounded}")
    print(f"Латентность p50={p50:.1f}s  p95={p95:.1f}s")


def _percentile(data: list[float], pct: float) -> float:
    if not data:
        return 0.0
    s = sorted(data)
    k = (len(s) - 1) * pct / 100
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


async def main() -> None:
    corpus = load_corpus()
    sweep = await sweep_thresholds(corpus)
    best = pick_threshold(sweep)
    await evaluate(corpus, best)


if __name__ == "__main__":
    asyncio.run(main())
