from __future__ import annotations

import asyncio

import pytest

from app.generation import service as gen_service
from app.schemas import ChunkResult


@pytest.fixture
def chunk() -> ChunkResult:
    return ChunkResult(
        chunk_id=1,
        content=(
            "Уровень 0: сервисы используют HTTP как транспорт, архитектура "
            "клиента-сервера, кэширование ответов и один HTTP-глагол POST."
        ),
        page=None,
        section="REST",
        source_name="Obsidian Vault/Resurses/REST/Code on done.md",
        score=0.7,
    )


class _FakeOllama:
    def __init__(self, reply: str) -> None:
        self._reply = reply

    async def chat(  # noqa: ANN001
        self,
        messages,
        *,
        temperature=0.0,
        options=None,
        seed=None,
        top_p=None,
        keep_alive=None,
    ):
        return self._reply


def test_bug_repro_marker_with_parenthetical_is_stub(monkeypatch, chunk) -> None:
    bug_reply = (
        "НЕВОЗМОЖНО_ОТВЕТИТЬ (отсутствуют сведения об архитектуре "
        "клиента-сервера, кэшировании и передачи кода)"
    )
    monkeypatch.setattr(gen_service, "ollama", _FakeOllama(bug_reply))

    result = asyncio.run(gen_service.answer("какая погода на марсе", [chunk]))

    assert result.grounded is False
    assert result.cited_chunk_ids == []
    assert result.answer == gen_service.STUB_ANSWER


def test_noanswer_sentinel_is_stub(monkeypatch, chunk) -> None:
    monkeypatch.setattr(gen_service, "ollama", _FakeOllama("NOANSWER"))

    result = asyncio.run(gen_service.answer("какая погода на марсе", [chunk]))
    assert result.grounded is False
    assert result.cited_chunk_ids == []


def test_grounded_answer_still_returns_source(monkeypatch, chunk) -> None:
    reply = "Сервисы используют HTTP как транспорт [c1]"
    monkeypatch.setattr(gen_service, "ollama", _FakeOllama(reply))

    result = asyncio.run(
        gen_service.answer("какой транспорт используют сервисы", [chunk])
    )
    assert result.grounded is True
    assert 1 in result.cited_chunk_ids
