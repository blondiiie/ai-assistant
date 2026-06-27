from __future__ import annotations

import asyncio

from app.db.session import engine

STATEMENTS = [
    "CREATE EXTENSION IF NOT EXISTS vector",
    "CREATE EXTENSION IF NOT EXISTS pg_trgm",
    """
    CREATE TABLE IF NOT EXISTS documents (
        id BIGSERIAL PRIMARY KEY,
        source_name TEXT NOT NULL,
        document_type VARCHAR(4) NOT NULL,
        version INT NOT NULL DEFAULT 1,
        active BOOLEAN NOT NULL DEFAULT TRUE,
        file_path TEXT NOT NULL,
        file_hash CHAR(64),
        file_mtime DOUBLE PRECISION,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ,
        deactivated_at TIMESTAMPTZ,
        CONSTRAINT ck_documents_type CHECK (document_type IN ('PDF','DOCX','TXT','MD'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS chunks (
        id BIGSERIAL PRIMARY KEY,
        document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE RESTRICT,
        chunk_index INT NOT NULL,
        content TEXT NOT NULL,
        page INT,
        section TEXT,
        embedding vector(768) NOT NULL,
        tsv tsvector GENERATED ALWAYS AS (to_tsvector('russian', content)) STORED NOT NULL,
        CONSTRAINT uq_chunks_doc_index UNIQUE (document_id, chunk_index)
    )
    """,
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_hash CHAR(64)",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS file_mtime DOUBLE PRECISION",
    "ALTER TABLE documents ADD COLUMN IF NOT EXISTS deactivated_at TIMESTAMPTZ",
    "ALTER TABLE documents ALTER COLUMN source_name TYPE TEXT",
    "ALTER TABLE documents DROP CONSTRAINT IF EXISTS ck_documents_type",
    "ALTER TABLE documents ADD CONSTRAINT ck_documents_type "
    "CHECK (document_type IN ('PDF','DOCX','TXT','MD'))",
    "CREATE INDEX IF NOT EXISTS ix_chunks_document_id ON chunks (document_id)",
    "CREATE INDEX IF NOT EXISTS ix_chunks_tsv ON chunks USING GIN (tsv)",
    """
    CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
    """,
    """
    CREATE TABLE IF NOT EXISTS document_links (
        id BIGSERIAL PRIMARY KEY,
        source_doc_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
        target_title TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS ix_document_links_source ON document_links (source_doc_id)",
    "CREATE INDEX IF NOT EXISTS ix_document_links_title ON document_links (target_title)",
    "CREATE INDEX IF NOT EXISTS ix_documents_active_source ON documents (active, source_name)",
]


async def main() -> None:
    async with engine.begin() as conn:
        for stmt in STATEMENTS:
            await conn.exec_driver_sql(stmt.strip())
    print("Схема инициализирована: extensions, documents, chunks, индексы.")


if __name__ == "__main__":
    asyncio.run(main())
