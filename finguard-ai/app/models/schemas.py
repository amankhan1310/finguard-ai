"""
app/models/schemas.py
─────────────────────
Pydantic v2 request / response schemas.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


# ── Health ────────────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    app: str
    version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Upload ────────────────────────────────────────────────────────────


class ChunkMeta(BaseModel):
    chunk_index: int
    total_chunks: int
    page: int | None = None


class UploadResponse(BaseModel):
    message: str
    filename: str
    chunks_indexed: int
    collection: str


# ── Ask / RAG ─────────────────────────────────────────────────────────


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def strip_whitespace(cls, v: str) -> str:
        stripped = v.strip()
        if not stripped:
            raise ValueError("Question must not be blank.")
        return stripped


class SourceDocument(BaseModel):
    filename: str
    page: int | None = None
    chunk_index: int | None = None
    excerpt: str | None = None  # first 200 chars of the chunk


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDocument]
    model_used: str


# ── Error ─────────────────────────────────────────────────────────────


class ErrorDetail(BaseModel):
    error: str
    detail: str
    status_code: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)
