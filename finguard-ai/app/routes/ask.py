"""
app/routes/ask.py
───────────────────
POST /ask — Receive a question, run the RAG pipeline, return the answer.
"""

from fastapi import APIRouter, Depends
from loguru import logger

from app.models.schemas import AskRequest, AskResponse
from app.services.query_service import QueryService

router = APIRouter(prefix="/api/v1", tags=["RAG"])


def _get_query_service() -> QueryService:
    return QueryService()


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a compliance or fraud investigation question",
    description=(
        "Submit a natural-language question. The RAG pipeline will retrieve "
        "relevant chunks from indexed documents and use Gemini 1.5 Flash to "
        "generate a grounded answer with source citations."
    ),
)
def ask_question(
    body: AskRequest,
    service: QueryService = Depends(_get_query_service),
) -> AskResponse:
    logger.info(f"Ask request: '{body.question[:80]}'")
    return service.answer(body)
