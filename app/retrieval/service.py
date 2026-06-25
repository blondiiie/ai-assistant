from __future__ import annotations

import httpx
from sqlalchemy import text

from app.config import settings
from app.db.session import async_session
from app.schemas import ChunkResult, RetrieveResult

CANDIDATE_SQL = text(
    """
    SELECT c.id AS chunk_id,
           c.content,
           c.page,
           c.section,
           d.source_name,
           1 - (c.embedding <=> (:q)::vector) AS sim,
           ts_rank(c.tsv, plainto_tsquery('russian', :qtext)) AS lex
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE d.active = true
    ORDER BY c.embedding <=> (:q)::vector
    LIMIT :candidates
    """
)


async def _embed_query(query: str) -> list[float]:
    async with httpx.AsyncClient(timeout=settings.embed_timeout) as client:
        resp = await client.post(
            f"{settings.ollama_url}/api/embed",
            json={"model": settings.embed_model, "input": [query]},
        )
    resp.raise_for_status()
    vectors = resp.json()["embeddings"]
    if not vectors:
        raise RuntimeError("empty embedding for query")
    return list(map(float, vectors[0]))


def _hybrid_score(rows: list[dict], alpha: float) -> list[dict]:
    max_sim = max((r["sim"] for r in rows), default=0.0) or 1.0
    max_lex = max((r["lex"] for r in rows), default=0.0) or 1e-9
    for r in rows:
        sim_norm = r["sim"] / max_sim
        lex_norm = r["lex"] / max_lex
        r["score"] = alpha * sim_norm + (1 - alpha) * lex_norm
    return rows


async def search(query: str, top_k: int | None = None) -> RetrieveResult:
    top_k = top_k or settings.top_k
    qvec = await _embed_query(query)

    async with async_session() as session:
        result = await session.execute(
            CANDIDATE_SQL,
            {
                "q": f"[{','.join(f'{x:.7f}' for x in qvec)}]",
                "qtext": query,
                "candidates": settings.retrieval_candidates,
            },
        )
        rows_raw = result.mappings().all()

    rows = [dict(r) for r in rows_raw]

    passing = [r for r in rows if r["sim"] >= settings.sim_threshold]
    if not passing:
        return RetrieveResult(found=False, results=[])

    scored = _hybrid_score(passing, settings.hybrid_alpha)
    scored.sort(key=lambda r: r["score"], reverse=True)
    top = scored[:top_k]

    results = [
        ChunkResult(
            chunk_id=r["chunk_id"],
            content=r["content"],
            page=r["page"],
            section=r["section"],
            source_name=r["source_name"],
            score=round(float(r["score"]), 4),
        )
        for r in top
    ]
    return RetrieveResult(found=True, results=results)
