from __future__ import annotations

from pathlib import Path

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Chunk, Document, DocumentLink
from app.db.session import async_session
from app.ingest.chunker import achunk_blocks
from app.ingest.parsers import extract_md_links, parse
from app.llm.client import ollama
from app.schemas import ChunkMeta

EMBED_BATCH = 16


async def parse_and_chunk(file_path: str, document_type: str) -> list[ChunkMeta]:
    blocks = parse(file_path, document_type)
    return await achunk_blocks(blocks)


async def _embed_chunks(metas: list[ChunkMeta]) -> list[list[float]]:
    vectors: list[list[float]] = []
    for i in range(0, len(metas), EMBED_BATCH):
        batch = metas[i : i + EMBED_BATCH]
        vectors.extend(await ollama.embed([m.content for m in batch]))
    return vectors


async def store(
    source_name: str,
    document_type: str,
    file_path: str,
    file_hash: str | None = None,
    file_mtime: float | None = None,
) -> tuple[int, int]:
    metas = await parse_and_chunk(file_path, document_type)
    vectors = await _embed_chunks(metas)

    async with async_session() as session:
        async with session.begin():
            version = await _next_version(session, source_name)
            await session.execute(
                update(Document)
                .where(Document.source_name == source_name, Document.active.is_(True))
                .values(active=False, updated_at=func.now())
            )
            doc = Document(
                source_name=source_name,
                document_type=document_type,
                version=version,
                active=True,
                file_path=str(Path(file_path).resolve()),
                file_hash=file_hash,
                file_mtime=file_mtime,
                deactivated_at=None,
            )
            session.add(doc)
            await session.flush()
            for meta, vector in zip(metas, vectors, strict=True):
                session.add(
                    Chunk(
                        document_id=doc.id,
                        chunk_index=meta.chunk_index,
                        content=meta.content,
                        page=meta.page,
                        section=meta.section,
                        embedding=vector,
                    )
                )
            if document_type == "MD":
                for target in extract_md_links(file_path):
                    session.add(
                        DocumentLink(source_doc_id=doc.id, target_title=target)
                    )
        return doc.id, len(metas)


async def _next_version(session: AsyncSession, source_name: str) -> int:
    result = await session.execute(
        select(Document.version)
        .where(Document.source_name == source_name)
        .order_by(Document.version.desc())
    )
    latest = result.scalars().first()
    return (latest or 0) + 1
