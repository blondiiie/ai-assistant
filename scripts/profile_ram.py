"""Профиль RAM RAG-пайплайна (Этап 4.3 рефакторинга).

Автономный скрипт: прогоняет N запросов через полный пайплайн (retrieval +
generation + grounding) и снимает метрики памяти после каждого. НЕ требует
ручного наблюдения — завершается сам с детерминированным отчётом и exit-code.

Защита от «зацикливания» (почему предыдущие проверки RAM не заканчивались):
  • жёсткий лимит итераций (--requests, дефолт 5);
  • жёсткий таймаут на каждый запрос asyncio.wait_for (--per-request-timeout);
  • жёсткий общий таймаут (SIGALRM/watchdog) — аварийный выход, если event loop
    всё же завис;
  • простые короткие вопросы, чтобы LLM отвечала быстро;
  • детерминированный exit-code (0 — нет утечек и пик в зелёной зоне).

Запуск:
    uv run python -m scripts.profile_ram
    uv run python -m scripts.profile_ram --requests 10 --per-request-timeout 180

Требует поднятых Postgres и Ollama (open -a OrbStack; ollama serve).
Результат также пишется в data/ram_profile.json.
"""
from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import signal
import sys
import time
from pathlib import Path

import psutil

from app.assistant import ask
from app.config import settings

logging.basicConfig(
    level=logging.WARNING,  # INFO от пайплайна не мешает таблице RAM
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("profile_ram")

OUT_JSON = Path(__file__).resolve().parent.parent / "data" / "ram_profile.json"

# MacBook Air 16 ГБ: зелёная зона для Python-процесса RAG.
# Боты держат веса в Ollama (отдельный процесс ~2 ГБ), Python лишь оркестрирует.
RAM_GREEN_ZONE_MB = 400
# Порог «утечки»: монотонный рост RSS от запроса к запросу, не стабилизирующийся.
LEAK_DELTA_MB = 10

DEFAULT_QUESTIONS = [
    "что такое JSON",
    "что такое HATEOAS",
    "расскажи всё про REST",
    "какие принципы REST",
    "уровни зрелости REST",
]


def _proc_rss_mb(proc: psutil.Process) -> float:
    """RSS процесса в МБ (physical memory)."""
    return proc.memory_info().rss / 1024 / 1024


def _system_ram_mb() -> tuple[float, float]:
    """(used_mb, total_mb) системной RAM."""
    vm = psutil.virtual_memory()
    return (vm.total - vm.available) / 1024 / 1024, vm.total / 1024 / 1024


def _ollama_proc_mb() -> float | None:
    """RSS процессов Ollama (веса модели), если найдены."""
    total = 0.0
    found = False
    for p in psutil.process_iter(["name", "memory_info"]):
        try:
            name = (p.info.get("name") or "").lower()
            if "ollama" in name:
                mi = p.info.get("memory_info")
                if mi:
                    total += mi.rss / 1024 / 1024
                    found = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return total if found else None


def _snapshot(label: str) -> dict:
    proc = psutil.Process()
    used, total = _system_ram_mb()
    return {
        "label": label,
        "python_rss_mb": round(_proc_rss_mb(proc), 1),
        "ollama_rss_mb": round(_ollama_proc_mb(), 1) if _ollama_proc_mb() is not None else None,
        "system_used_mb": round(used, 1),
        "system_total_mb": round(total, 1),
        "system_used_pct": round(used / total * 100, 1),
    }


def _print_row(idx: int, snap: dict, grounded: bool, dt: float) -> None:
    ollama = f"{snap['ollama_rss_mb']:.0f}MB" if snap["ollama_rss_mb"] is not None else "n/a"
    print(
        f"  [{idx}] py={snap['python_rss_mb']:6.1f}MB  ollama={ollama:>8}  "
        f"sys={snap['system_used_pct']:4.1f}%  "
        f"grounded={'да' if grounded else 'НЕТ':3}  t={dt:5.1f}s"
    )


class _Watchdog:
    """Жёсткий общий таймаут через SIGALRM — аварийный выход, если event loop завис.

    SIGALRM работает только в main thread Unix — достаточно для dev-скрипта на macOS.
    """

    def __init__(self, seconds: float) -> None:
        self.seconds = seconds

    def _handler(self, *args: object) -> None:
        print(
            f"\n[WATCHDOG] Превышен общий таймаут {self.seconds}s — аварийное завершение.",
            file=sys.stderr,
        )
        sys.exit(124)

    def __enter__(self) -> None:
        try:
            signal.signal(signal.SIGALRM, self._handler)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        except (ValueError, AttributeError):
            # Не main thread или Windows — пропускаем; per-request таймаут всё равно ловит.
            pass

    def __exit__(self, *args: object) -> None:
        with contextlib.suppress(ValueError, AttributeError):
            signal.setitimer(signal.ITIMER_REAL, 0)


async def profile(questions: list[str], per_request_timeout: float) -> list[dict]:
    rows: list[dict] = []
    snap0 = _snapshot("baseline")
    print(f"baseline: python_rss={snap0['python_rss_mb']:.1f}MB "
          f"ollama={snap0['ollama_rss_mb']}MB "
          f"sys={snap0['system_used_pct']}%")

    for i, q in enumerate(questions, 1):
        snap_before = _snapshot(f"before#{i}")
        start = time.perf_counter()
        try:
            outcome = await asyncio.wait_for(ask(q), timeout=per_request_timeout)
            grounded = outcome.grounded
            err = None
        except TimeoutError:
            grounded = False
            err = f"timeout ({per_request_timeout}s)"
        except Exception as exc:  # noqa: BLE001
            grounded = False
            err = f"{type(exc).__name__}: {exc}"
        dt = time.perf_counter() - start
        snap_after = _snapshot(f"after#{i}")
        snap_after["grounded"] = grounded
        snap_after["error"] = err
        snap_after["dt_s"] = round(dt, 1)
        snap_after["delta_rss_mb"] = round(
            snap_after["python_rss_mb"] - snap_before["python_rss_mb"], 1
        )
        _print_row(i, snap_after, grounded, dt)
        rows.append(snap_after)
    return rows


def _verdict(rows: list[dict], baseline: dict) -> tuple[bool, str]:
    """(ok, сообщение) — нет утечек и пик в зелёной зоне."""
    if not rows:
        return False, "нет данных"
    rss_values = [r["python_rss_mb"] for r in rows]
    peak = max(rss_values)
    last = rss_values[-1]
    # Утечка: рост между первым и последним, причём не стабилизировался
    # (последние 2 запроса продолжают расти).
    grew = last - rss_values[0]
    tail_growing = len(rss_values) >= 3 and rss_values[-1] > rss_values[-2] > rss_values[-3]
    leak = grew > LEAK_DELTA_MB and tail_growing
    if peak > RAM_GREEN_ZONE_MB:
        return False, (
            f"ПИК {peak:.0f}MB > зелёной зоны {RAM_GREEN_ZONE_MB}MB — "
            "рассмотри 3B->более лёгкая модель или уменьшение top_k/ctx"
        )
    if leak:
        return False, (
            f"подозрение на УТЕЧКУ: рост RSS {rss_values[0]:.0f}->{last:.0f}MB "
            f"(+{grew:.0f}MB), хвост монотонно растёт"
        )
    return True, (
        f"OK: пик {peak:.0f}MB <= {RAM_GREEN_ZONE_MB}MB, "
        f"рост {rss_values[0]:.0f}->{last:.0f}MB (стабильно)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Профиль RAM RAG-пайплайна (Этап 4.3)")
    parser.add_argument(
        "--requests", type=int, default=5,
        help="Число запросов (дефолт 5)",
    )
    parser.add_argument(
        "--per-request-timeout", type=float, default=180.0,
        help="Таймаут на один запрос в секундах (дефолт 180)",
    )
    parser.add_argument(
        "--total-timeout", type=float, default=1800.0,
        help="Общий watchdog-таймаут в секундах (дефолт 1800=30мин)",
    )
    parser.add_argument(
        "--question", type=str, default=None,
        help="Один вопрос вместо стандартного набора (полезно для воспроизведения)",
    )
    args = parser.parse_args()

    if args.question:
        questions = [args.question] * args.requests
    else:
        questions = DEFAULT_QUESTIONS[: args.requests]

    print(f"=== Профиль RAM: {len(questions)} запросов | "
          f"модель={settings.llm_model} num_ctx={settings.llm_num_ctx} ===")
    print(f"зелёная зона Python-RSS: <= {RAM_GREEN_ZONE_MB}MB\n")

    with _Watchdog(args.total_timeout):
        try:
            rows = asyncio.run(profile(questions, args.per_request_timeout))
        except KeyboardInterrupt:
            print("\nПрервано пользователем.", file=sys.stderr)
            sys.exit(130)

    baseline = _snapshot("baseline")
    ok, msg = _verdict(rows, baseline)
    print(f"\nВердикт: {'✅ ' + msg if ok else '🔴 ' + msg}")

    report = {
        "model": settings.llm_model,
        "num_ctx": settings.llm_num_ctx,
        "keep_alive": settings.llm_keepalive,
        "requests": len(questions),
        "green_zone_mb": RAM_GREEN_ZONE_MB,
        "baseline": baseline,
        "rows": rows,
        "verdict_ok": ok,
        "verdict_msg": msg,
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Отчёт: {OUT_JSON}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()