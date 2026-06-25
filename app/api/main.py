from __future__ import annotations

from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.ingest.parsers import SUPPORTED_EXTENSIONS
from app.ingest.service import store
from app.schemas import ErrorResponse, UploadResponse


def create_app() -> FastAPI:
    app = FastAPI(title="Корпоративный AI-ассистент", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post(
        "/documents",
        response_model=UploadResponse,
        responses={400: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def upload_document(
        file: Annotated[UploadFile, File(description="Документ")],
    ) -> JSONResponse:
        ext = (Path(file.filename or "").suffix or "").lower().lstrip(".")
        doc_type = SUPPORTED_EXTENSIONS.get(ext)
        if doc_type is None:
            raise HTTPException(status_code=400, detail=f"Неподдерживаемый формат: .{ext}")

        uploads = Path(settings.uploads_dir)
        uploads.mkdir(parents=True, exist_ok=True)
        dest = uploads / f"{file.filename}"
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Пустой файл")
        if len(content) > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(
                status_code=413, detail=f"Файл больше {settings.max_upload_mb} МБ"
            )
        with open(dest, "wb") as f:
            f.write(content)

        name = file.filename or dest.name
        try:
            document_id, chunks_created = await store(name, doc_type, str(dest))
        except ValueError as exc:
            dest.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc

        return JSONResponse(
            status_code=200,
            content=UploadResponse(
                document_id=document_id,
                source_name=file.filename or dest.name,
                document_type=doc_type,
                chunks_created=chunks_created,
                status="indexed",
            ).model_dump(),
        )

    return app


app = create_app()
