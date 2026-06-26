from __future__ import annotations

from types import SimpleNamespace

from app.bot.main import _format


def _src(name: str, section: str | None = None, page: int | None = None):
    return SimpleNamespace(source_name=name, section=section, page=page)


def test_format_strips_citations_and_adds_source() -> None:
    outcome = SimpleNamespace(
        answer="JSON [c10] это формат [c9] обмена данными.",
        grounded=True,
        sources=[_src("vault/JSON.md", section="Основное")],
    )
    out = _format(outcome)
    assert "[c10]" not in out
    assert "[Источник: vault/JSON.md · раздел «Основное»]" in out


def test_format_dedupes_sources() -> None:
    outcome = SimpleNamespace(
        answer="текст [c1]",
        grounded=True,
        sources=[_src("a.md", "S"), _src("a.md", "S")],
    )
    out = _format(outcome)
    assert out.count("[Источник:") == 1


def test_format_stub_without_sources() -> None:
    outcome = SimpleNamespace(answer="Нет данных.", grounded=False, sources=[])
    assert _format(outcome) == "Нет данных."


def test_format_fallback_page_when_no_section() -> None:
    outcome = SimpleNamespace(
        answer="текст [c2]",
        grounded=True,
        sources=[_src("doc.pdf", section=None, page=4)],
    )
    out = _format(outcome)
    assert "стр. 4" in out
