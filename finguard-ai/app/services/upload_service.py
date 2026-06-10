"""
app/services/upload_service.py
────────────────────────────────
Handles everything related to PDF ingestion:

  1. Validate file type and size
  2. Persist the file to UPLOAD_DIR
  3. Delegate to document_processor for chunking
  4. Delegate to vector_store for embedding + ChromaDB storage
  5. Return an UploadResponse
"""

from pathlib import Path

import aiofiles
from fastapi import UploadFile
from loguru import logger

from app.core.config import get_settings
from app.core.exceptions import FileTooLargeError, InvalidFileTypeError
from app.models.schemas import UploadResponse
from app.rag.document_processor import load_and_chunk_pdf
from app.rag.vector_store import get_vector_store

_ALLOWED_CONTENT_TYPES = {"application/pdf"}
_ALLOWED_EXTENSIONS = {".pdf"}


class UploadService:
    """Orchestrates PDF upload → chunk → embed → store."""

    def __init__(self) -> None:
        self._settings = get_settings()

    # ── Public ─────────────────────────────────────────────────────────

    async def ingest(self, file: UploadFile) -> UploadResponse:
        """
        Full ingestion pipeline for an uploaded PDF.

        Parameters
        ----------
        file : UploadFile
            The multipart file received by FastAPI.

        Returns
        -------
        UploadResponse
            Summary of the ingestion result.

        Raises
        ------
        InvalidFileTypeError, FileTooLargeError, EmptyPDFError
        """
        self._validate_file_type(file)
        saved_path = await self._save_file(file)
        chunks = load_and_chunk_pdf(saved_path)

        vector_store = get_vector_store()
        vector_store.add_documents(chunks)

        logger.info(
            f"Ingestion complete: '{file.filename}' → {len(chunks)} chunk(s) indexed."
        )

        return UploadResponse(
            message="PDF successfully processed and indexed.",
            filename=file.filename or saved_path.name,
            chunks_indexed=len(chunks),
            collection=self._settings.chroma_collection_name,
        )

    # ── Private helpers ─────────────────────────────────────────────────

    def _validate_file_type(self, file: UploadFile) -> None:
        """Raise InvalidFileTypeError if the file is not a PDF."""
        filename = file.filename or ""
        suffix = Path(filename).suffix.lower()
        content_type = (file.content_type or "").lower()

        if suffix not in _ALLOWED_EXTENSIONS or content_type not in _ALLOWED_CONTENT_TYPES:
            raise InvalidFileTypeError(filename)

    async def _save_file(self, file: UploadFile) -> Path:
        """
        Stream the upload to disk and enforce the max-size limit.

        Returns
        -------
        Path
            Absolute path of the saved file.
        """
        dest: Path = self._settings.upload_dir / (file.filename or "upload.pdf")
        dest.parent.mkdir(parents=True, exist_ok=True)

        max_bytes = self._settings.max_upload_size_bytes
        bytes_written = 0

        logger.info(f"Saving upload to: {dest}")
        async with aiofiles.open(dest, "wb") as out_file:
            while chunk := await file.read(1024 * 64):  # 64 KB chunks
                bytes_written += len(chunk)
                if bytes_written > max_bytes:
                    await out_file.close()
                    dest.unlink(missing_ok=True)
                    raise FileTooLargeError(self._settings.max_upload_size_mb)
                await out_file.write(chunk)

        logger.info(f"  → Saved {bytes_written:,} bytes")
        return dest
