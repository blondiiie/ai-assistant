from __future__ import annotations

from pathlib import Path

from app.ingest.parsers import extract_md_links, parse_md


def test_md_strips_frontmatter(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_text(
        "---\ntags: [json]\ndate: 2024-01-01\n---\n\n"
        "# Заголовок\n\nТекст заметки без метаданных.",
        encoding="utf-8",
    )
    blocks = parse_md(str(f))
    assert blocks
    joined = "\n".join(b.text for b in blocks)
    assert "tags" not in joined
    assert "Текст заметки" in joined


def test_md_unfolds_wikilinks(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_text(
        "См. [[JSON-объект]] и [[REST|REST API]] подробнее.", encoding="utf-8"
    )
    blocks = parse_md(str(f))
    text = blocks[0].text
    assert "JSON-объект" in text
    assert "REST API" in text
    assert "[[" not in text


def test_md_headings_become_sections(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_text(
        "# Главный\n\nвступление\n\n## Подраздел\n\nдетали\n", encoding="utf-8"
    )
    blocks = parse_md(str(f))
    sections = [b.section for b in blocks if b.section]
    assert "Главный" in sections
    assert "Подраздел" in sections


def test_extract_md_links_targets(tmp_path: Path) -> None:
    f = tmp_path / "REST API.md"
    f.write_text(
        "---\ntags: [rest]\n---\n"
        "Принципы:\n1) [[Stateless]]\n2) [[Кэширование]]\n"
        "См. [[REST|REST API]] и блок [[REST#^9f9975]].\n",
        encoding="utf-8",
    )
    links = extract_md_links(str(f))
    assert links == ["Stateless", "Кэширование", "REST"]


def test_extract_md_links_empty_when_none(tmp_path: Path) -> None:
    f = tmp_path / "note.md"
    f.write_text("Обычный текст без ссылок.", encoding="utf-8")
    assert extract_md_links(str(f)) == []
