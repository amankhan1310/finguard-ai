"""
app/services/query_service.py
───────────────────────────────
Thin service layer that sits between the /ask route and the RAGChain.
Keeping this separate allows future additions (e.g. query logging,
rate limiting, caching) without touching the route or RAG layer.
"""

from loguru import logger

from app.core.exceptions import EmptyQuestionError
from app.models.schemas import AskRequest, AskResponse
from app.rag.rag_chain import RAGChain

# Module-level singleton — created once per worker process
_rag_chain: RAGChain | None = None


def _get_rag_chain() -> RAGChain:
    global _rag_chain
    if _rag_chain is None:
        _rag_chain = RAGChain()
    return _rag_chain


class QueryService:
    """Handles the question → RAG → answer flow."""

    def answer(self, request: AskRequest) -> AskResponse:
        """
        Process a question and return a grounded answer with sources.

        Parameters
        ----------
        request : AskRequest
            Validated Pydantic model containing the user's question.

        Returns
        -------
        AskResponse
            Answer text + source documents.

        Raises
        ------
        EmptyQuestionError, NoDocumentsIndexedError, GeminiServiceError
        """
        if not request.question.strip():
            raise EmptyQuestionError()

        logger.info(f"Processing question: '{request.question[:80]}'")
        chain = _get_rag_chain()
        return chain.run(request.question)
