"""
app/routes/health.py
──────────────────────
GET /health — liveness probe.
"""

from datetime import datetime

from fastapi import APIRouter

from app.core.config import get_settings
from app.models.schemas import HealthResponse
from app.rag.vector_store import get_vector_store

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health_check() -> HealthResponse:
    """
    Returns application status, version, and the number of indexed chunks.
    ChromaDB count is included as a quick sanity check.
    """
    settings = get_settings()
    vs = get_vector_store()
    indexed = vs.collection_count()

    return HealthResponse(
        status="ok",
        app=settings.app_name,
        version=settings.app_version,
        timestamp=datetime.utcnow(),
        # Extra field added dynamically (Pydantic v2 allows model_extra)
    ).model_copy(
        update={"indexed_chunks": indexed}  # surfaced in JSON even without schema field
    )
