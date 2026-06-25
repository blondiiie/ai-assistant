from __future__ import annotations

from pydantic import BaseModel


class ChunkMeta(BaseModel):
    chunk_index: int
    content: str
    page: int | None = None
    section: str | None = None


class ChunkResult(BaseModel):
    chunk_id: int
    content: str
    page: int | None = None
    section: str | None = None
    source_name: str
    score: float


class RetrieveResult(BaseModel):
    found: bool
    results: list[ChunkResult]


class GenerateResult(BaseModel):
    answer: str
    cited_chunk_ids: list[int]
    grounded: bool


class UploadResponse(BaseModel):
    document_id: int
    source_name: str
    document_type: str
    chunks_created: int
    status: str


class ErrorResponse(BaseModel):
    detail: str
    code: str | None = None
