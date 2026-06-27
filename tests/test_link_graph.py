from __future__ import annotations

from app.retrieval.service import _resolve_neighbor_doc_ids, _stem


def test_stem_strips_dir_and_ext() -> None:
    assert _stem("Obsidian Vault/Resurses/REST/REST API.md") == "REST API"
    assert _stem("JSON.md") == "JSON"


def test_resolve_outgoing_and_incoming_neighbors() -> None:
    # REST API (id 1) ссылается на Stateless, Кэширование, JSON-объект.
    # Stateless (id 2), Кэширование (id 3), JSON (id 4) ссылается на JSON-объект (id 5).
    active_docs = [
        (1, "Obsidian Vault/Resurses/REST/REST API.md"),
        (2, "Obsidian Vault/Resurses/REST/Stateless.md"),
        (3, "Obsidian Vault/Resurses/REST/Кэширование.md"),
        (4, "Obsidian Vault/Resurses/JSON/JSON.md"),
        (5, "Obsidian Vault/Resurses/JSON/JSON-объект.md"),
    ]
    all_links = [
        (1, "Stateless"),
        (1, "Кэширование"),
        (4, "JSON-объект"),
    ]

    # Выбрали REST API -> соседи: Stateless, Кэширование (исходящие).
    neighbors = _resolve_neighbor_doc_ids(
        relevant_doc_ids={1},
        relevant_stems={"REST API"},
        active_docs=active_docs,
        all_links=all_links,
    )
    assert set(neighbors) == {2, 3}

    # Выбрали JSON-объект -> входящая ссылка из JSON (id 4).
    neighbors_in = _resolve_neighbor_doc_ids(
        relevant_doc_ids={5},
        relevant_stems={"JSON-объект"},
        active_docs=active_docs,
        all_links=all_links,
    )
    assert neighbors_in == [4]

    # Неразрешённые ссылки игнорируются.
    assert _resolve_neighbor_doc_ids({1}, {"REST API"}, active_docs, [(1, "Несуществует")]) == []
