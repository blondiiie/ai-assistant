from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path

from sqlalchemy import delete, func, select, update

from app.config import settings
from app.db.models import Chunk, Document
from app.db.session import async_session
from app.ingest.parsers import SUPPORTED_EXTENSIONS
from app.ingest.service import store


@dataclass
class ScanResult:
    indexed: int = 0
    reindexed: int = 0
    skipped: int = 0
    deactivated: int = 0
    gc_deleted: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"новых: {self.indexed}, обновлено: {self.reindexed}, "
            f"без изменений: {self.skipped}, деактивировано: {self.deactivated}, "
            f"удалено(GC): {self.gc_deleted}"
            + (f", ошибок: {len(self.errors)}" if self.errors else "")
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _iter_files(root: Path):
    ignore = settings.ignore_set
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in ignore or part.startswith(".") for part in rel_parts):
            continue
        ext = path.suffix.lower().lstrip(".")
        if ext not in SUPPORTED_EXTENSIONS:
            continue
        yield path


async def _active_docs() -> dict[str, str | None]:
    async with async_session() as session:
        result = await session.execute(
            select(Document.source_name, Document.file_hash).where(Document.active.is_(True))
        )
        return {source_name: file_hash for source_name, file_hash in result.all()}


async def _index(source_name, path, doc_type, file_hash, result: ScanResult, *, changed: bool):
    try:
        await store(source_name, doc_type, str(path), file_hash=file_hash)
    except (ValueError, RuntimeError) as exc:
        result.errors.append(f"{source_name}: {exc}")
        return
    if changed:
        result.reindexed += 1
    else:
        result.indexed += 1


async def _gc() -> int:
    cutoff = datetime.now(UTC) - timedelta(days=settings.gc_retention_days)
    async with async_session() as session, session.begin():
        ids = (
            await session.execute(
                select(Document.id).where(
                    Document.active.is_(False),
                    Document.deactivated_at.is_not(None),
                    Document.deactivated_at < cutoff,
                )
            )
        ).scalars().all()
        if not ids:
            return 0
        await session.execute(delete(Chunk).where(Chunk.document_id.in_(ids)))
        await session.execute(delete(Document).where(Document.id.in_(ids)))
        return len(ids)


async def scan() -> ScanResult:
    result = ScanResult()
    on_disk: dict[str, tuple[Path, str, str]] = {}

    for root_str in settings.source_list:
        root = Path(root_str).expanduser()
        if not root.exists():
            result.errors.append(f"источник не найден: {root}")
            continue
        root_name = root.name
        for path in _iter_files(root):
            ext = path.suffix.lower().lstrip(".")
            doc_type = SUPPORTED_EXTENSIONS[ext]
            rel = path.relative_to(root).as_posix()
            source_name = f"{root_name}/{rel}"
            try:
                digest = _sha256(path)
            except OSError as exc:
                result.errors.append(f"{source_name}: {exc}")
                continue
            on_disk[source_name] = (path, doc_type, digest)

    active = await _active_docs()

    for source_name, (path, doc_type, digest) in on_disk.items():
        existing = active.get(source_name)
        if existing is None:
            await _index(source_name, path, doc_type, digest, result, changed=False)
        elif existing != digest:
            await _index(source_name, path, doc_type, digest, result, changed=True)
        else:
            result.skipped += 1

    missing = [sn for sn in active if sn not in on_disk]
    if missing:
        async with async_session() as session, session.begin():
            await session.execute(
                update(Document)
                .where(Document.source_name.in_(missing), Document.active.is_(True))
                .values(active=False, deactivated_at=func.now())
            )
        result.deactivated = len(missing)

    result.gc_deleted = await _gc()
    return result
