from __future__ import annotations

import re
from pathlib import PurePosixPath

import httpx
from sqlalchemy import bindparam, text

from app.config import settings
from app.db.session import async_session
from app.schemas import ChunkResult, RetrieveResult

_TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
_TOPIC_STOP = {
    "что", "какой", "какая", "какие", "какое", "чем", "всё", "все", "всех",
    "мне", "дай", "дайте", "расскажи", "расскажите", "опиши", "опишите",
    "подробнее", "определение", "отличие", "разница", "различие", "похожи",
    "общее", "общего", "сколько", "когда", "где", "почему", "зачем", "это",
    "этом", "такое", "таком", "про", "о", "об", "для", "при", "между",
}

CANDIDATE_SQL = text(
    """
    SELECT c.id AS chunk_id,
           c.content,
           c.page,
           c.section,
           d.id AS document_id,
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

ACTIVE_DOCS_SQL = text("SELECT id, source_name FROM documents WHERE active = true")

ALL_LINKS_SQL = text("SELECT source_doc_id, target_title FROM document_links")

NEIGHBOR_CHUNKS_SQL = text(
    """
    SELECT c.id AS chunk_id,
           c.content,
           c.page,
           c.section,
           d.id AS document_id,
           d.source_name,
           1 - (c.embedding <=> (:q)::vector) AS sim,
           ts_rank(c.tsv, plainto_tsquery('russian', :qtext)) AS lex
    FROM chunks c
    JOIN documents d ON d.id = c.document_id
    WHERE d.active = true AND d.id IN :doc_ids
    """
).bindparams(bindparam("doc_ids", expanding=True))


def _stem(source_name: str) -> str:
    """basename без расширения: 'a/b/REST API.md' -> 'REST API'."""
    return PurePosixPath(source_name).stem


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


def _query_keywords(query: str) -> list[str]:
    keywords = []
    for tok in _TOKEN_RE.findall(query.lower()):
        if tok.isdigit() or (len(tok) >= 3 and tok not in _TOPIC_STOP):
            keywords.append(tok)
    return keywords


def _topic_match(source_name: str, section: str | None, keyword: str) -> bool:
    hay = f"{source_name} {section or ''}".lower()
    return keyword in hay


def _apply_topic_gate(rows: list[dict], query: str) -> list[dict]:
    keywords = _query_keywords(query)
    if not keywords:
        return rows
    matched = [
        r for r in rows
        if any(_topic_match(r["source_name"], r["section"], kw) for kw in keywords)
    ]
    return matched or rows


def _resolve_neighbor_doc_ids(
    relevant_doc_ids: set[int],
    relevant_stems: set[str],
    active_docs: list[tuple[int, str]],
    all_links: list[tuple[int, str]],
) -> list[int]:
    """Документы-соседи по графу wikilinks (исходящие + входящие), не входящие в выборку.

    - исходящие: заметки, на которые ссылается выбранная заметка (target_title == stem соседа);
    - входящие: заметки, которые ссылаются на выбранную (target_title == stem выбранной).
    """
    stem_to_id = {_stem(name): did for did, name in active_docs}
    out_by_src: dict[int, list[str]] = {}
    in_links: dict[str, list[int]] = {}
    for src_id, title in all_links:
        out_by_src.setdefault(src_id, []).append(title)
        in_links.setdefault(title, []).append(src_id)

    neighbor_ids: set[int] = set()
    # исходящие из выбранных
    for did in relevant_doc_ids:
        for title in out_by_src.get(did, []):
            nid = stem_to_id.get(title)
            if nid is not None and nid not in relevant_doc_ids:
                neighbor_ids.add(nid)
    # входящие в выбранные
    for stem in relevant_stems:
        for nid in in_links.get(stem, []):
            if nid not in relevant_doc_ids:
                neighbor_ids.add(nid)
    return list(neighbor_ids)


async def _neighbor_chunks(
    doc_ids: list[int],
    qvec: list[float],
    query: str,
    vec_str: str,
) -> list[dict]:
    if not doc_ids:
        return []
    async with async_session() as session:
        result = await session.execute(
            NEIGHBOR_CHUNKS_SQL,
            {"doc_ids": doc_ids, "q": vec_str, "qtext": query},
        )
        return [dict(r) for r in result.mappings().all()]


async def search(query: str, top_k: int | None = None) -> RetrieveResult:
    top_k = top_k or settings.top_k
    qvec = await _embed_query(query)
    vec_str = f"[{','.join(f'{x:.7f}' for x in qvec)}]"

    async with async_session() as session:
        result = await session.execute(
            CANDIDATE_SQL,
            {
                "q": vec_str,
                "qtext": query,
                "candidates": settings.retrieval_candidates,
            },
        )
        rows_raw = result.mappings().all()

        active_docs: list[tuple[int, str]] = []
        all_links: list[tuple[int, str]] = []
        if settings.link_expansion:
            docs_res = await session.execute(ACTIVE_DOCS_SQL)
            active_docs = [(r.id, r.source_name) for r in docs_res.all()]
            links_res = await session.execute(ALL_LINKS_SQL)
            all_links = [(r.source_doc_id, r.target_title) for r in links_res.all()]

    rows = [dict(r) for r in rows_raw]

    passing = [r for r in rows if r["sim"] >= settings.sim_threshold]
    if not passing:
        return RetrieveResult(found=False, results=[])

    max_sim = max(r["sim"] for r in passing)
    gapped = [r for r in passing if r["sim"] >= max_sim - settings.sim_drop]

    relevant = _apply_topic_gate(gapped, query)

    scored = _hybrid_score(relevant, settings.hybrid_alpha)
    scored.sort(key=lambda r: r["score"], reverse=True)
    top = scored[:top_k]
    core_ids = {r["chunk_id"] for r in top}

    if settings.link_expansion and top:
        top_doc_ids = {r["document_id"] for r in top}
        top_stems = {_stem(r["source_name"]) for r in top}
        neighbor_ids = _resolve_neighbor_doc_ids(
            top_doc_ids, top_stems, active_docs, all_links
        )
        extra = await _neighbor_chunks(neighbor_ids, qvec, query, vec_str)
        if extra:
            extra = [r for r in extra if r["chunk_id"] not in core_ids]
            extra.sort(key=lambda r: r["sim"], reverse=True)
            top = top + extra[: settings.link_expansion_chunks]

    final = _hybrid_score(top, settings.hybrid_alpha)

    results = [
        ChunkResult(
            chunk_id=r["chunk_id"],
            content=r["content"],
            page=r["page"],
            section=r["section"],
            source_name=r["source_name"],
            score=round(float(r["score"]), 4),
        )
        for r in final
    ]
    return RetrieveResult(found=True, results=results)
