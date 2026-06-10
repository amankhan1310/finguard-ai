"""
app/rag/vector_store.py
────────────────────────
Thin, singleton-safe wrapper around ChromaDB + GoogleGenerativeAIEmbeddings.

Responsibilities
----------------
* Initialise (once) the embedding model and the persistent Chroma collection.
* Provide add_documents() and as_retriever() helpers used by other services.
* Expose collection_count() for health / debug purposes.
"""

from functools import lru_cache

import chromadb
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.exceptions import ChromaServiceError


class VectorStore:
    """Wrapper around a persistent ChromaDB collection."""

    def __init__(self) -> None:
        settings = get_settings()

        logger.info("Initialising embedding model …")
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )

        logger.info(f"Connecting to ChromaDB at '{settings.chroma_persist_dir}' …")
        try:
            self._chroma = Chroma(
                collection_name=settings.chroma_collection_name,
                embedding_function=self._embeddings,
                persist_directory=settings.chroma_persist_dir,
            )
        except Exception as exc:
            raise ChromaServiceError(str(exc)) from exc

        logger.info(
            f"ChromaDB ready — collection '{settings.chroma_collection_name}', "
            f"{self.collection_count()} document(s) already indexed."
        )

    # ── Public API ─────────────────────────────────────────────────────

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def add_documents(self, documents: list[Document]) -> None:
        """
        Embed and persist *documents* to ChromaDB.
        Retries up to 3 times on transient embedding/network failures.
        """
        if not documents:
            logger.warning("add_documents called with empty list — skipping.")
            return
        try:
            self._chroma.add_documents(documents)
            logger.info(f"  → {len(documents)} chunk(s) added to ChromaDB.")
        except Exception as exc:
            raise ChromaServiceError(str(exc)) from exc

    def as_retriever(self, top_k: int | None = None):
        """Return a LangChain VectorStoreRetriever for similarity search."""
        settings = get_settings()
        k = top_k or settings.retriever_top_k
        return self._chroma.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def collection_count(self) -> int:
        """Return total number of stored chunks."""
        try:
            return self._chroma._collection.count()
        except Exception:
            return -1


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Return the application-wide singleton VectorStore."""
    return VectorStore()
