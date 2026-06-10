"""
app/routes/upload.py
──────────────────────
POST /upload — Accept a PDF, run the full ingestion pipeline.
"""

from fastapi import APIRouter, Depends, File, UploadFile
from loguru import logger

from app.models.schemas import UploadResponse
from app.services.upload_service import UploadService

router = APIRouter(prefix="/api/v1", tags=["Documents"])


def _get_upload_service() -> UploadService:
    return UploadService()


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=201,
    summary="Upload and index a PDF document",
    description=(
        "Upload a PDF file. The server will extract text, split it into chunks, "
        "generate embeddings using Google's embedding model, and store them in "
        "ChromaDB for later retrieval."
    ),
)
async def upload_document(
    file: UploadFile = File(..., description="PDF file to ingest"),
    service: UploadService = Depends(_get_upload_service),
) -> UploadResponse:
    logger.info(f"Upload request received: '{file.filename}' ({file.content_type})")
    return await service.ingest(file)
