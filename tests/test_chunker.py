from __future__ import annotations

from app.ingest.chunker import _split_text, chunk_blocks
from app.ingest.parsers import TextBlock


def test_split_text_respects_size_and_overlap() -> None:
    text = "слово " * 2000
    pieces = _split_text(text, size=512, overlap=50)
    assert len(pieces) > 1
    for piece in pieces:
        assert piece.strip()


def test_split_text_empty() -> None:
    assert _split_text("", 512, 50) == []


def test_chunk_blocks_assigns_metadata_and_index() -> None:
    blocks = [
        TextBlock(text="альфа " * 600, page=1, section="Введение"),
        TextBlock(text="бета " * 600, page=2, section=None),
    ]
    metas = chunk_blocks(blocks, chunk_size=512, chunk_overlap=50)
    assert len(metas) > 1
    assert [m.chunk_index for m in metas] == list(range(len(metas)))
    assert metas[0].page == 1
    assert metas[0].section == "Введение"
    assert metas[-1].page == 2
    assert all(m.content.strip() for m in metas)
