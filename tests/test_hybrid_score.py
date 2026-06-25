from __future__ import annotations

from app.retrieval.service import _hybrid_score


def test_hybrid_score_normalizes_and_ranks() -> None:
    rows = [
        {"sim": 0.9, "lex": 0.0},
        {"sim": 0.6, "lex": 0.5},
        {"sim": 0.3, "lex": 1.0},
    ]
    out = _hybrid_score([dict(r) for r in rows], alpha=0.6)
    scores = [r["score"] for r in out]
    assert all(0.0 <= s <= 1.0 for s in scores)
    assert scores[0] == max(scores)


def test_hybrid_score_alpha_weights() -> None:
    rows = [
        {"sim": 1.0, "lex": 0.0},
        {"sim": 0.0, "lex": 1.0},
    ]
    sim_heavy = _hybrid_score([dict(r) for r in rows], alpha=0.9)
    assert sim_heavy[0]["score"] > sim_heavy[1]["score"]
    lex_heavy = _hybrid_score([dict(r) for r in rows], alpha=0.1)
    assert lex_heavy[1]["score"] > lex_heavy[0]["score"]
