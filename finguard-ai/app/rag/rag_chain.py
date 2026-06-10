"""
app/rag/rag_chain.py
─────────────────────
Builds and executes the retrieval-augmented generation pipeline:

  1. Retrieve top-K relevant chunks from ChromaDB
  2. Format them into a structured prompt
  3. Call Gemini 1.5 Flash for an answer
  4. Return the answer + deduplicated source documents
"""

from langchain.schema import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from app.core.config import get_settings
from app.core.exceptions import GeminiServiceError, NoDocumentsIndexedError
from app.models.schemas import AskResponse, SourceDocument
from app.rag.vector_store import get_vector_store


# ── Prompt template ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You are FinGuard AI, a specialist financial compliance and fraud investigation assistant.
Answer the user's question using ONLY the context provided below.
If the context does not contain enough information to answer, say:
"I could not find a relevant answer in the indexed documents."

Rules:
- Be precise and cite specific regulations or thresholds when present.
- Never fabricate facts or reference documents not in the context.
- Structure your answer clearly; use bullet points for multi-part answers.

Context:
{context}
"""

_PROMPT = ChatPromptTemplate.from_messages(
    [
        ("system", _SYSTEM_PROMPT),
        ("human", "{question}"),
    ]
)


# ── Helpers ────────────────────────────────────────────────────────────────────


def _format_docs(docs: list[Document]) -> str:
    """Concatenate chunk texts with document separators for the prompt."""
    sections = []
    for doc in docs:
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page", "?")
        sections.append(f"[Source: {filename}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(sections)


def _build_source_documents(docs: list[Document]) -> list[SourceDocument]:
    """Convert retrieved LangChain Documents into SourceDocument schema objects."""
    seen: set[tuple] = set()
    sources: list[SourceDocument] = []
    for doc in docs:
        filename = doc.metadata.get("filename", "unknown")
        page = doc.metadata.get("page")
        chunk_index = doc.metadata.get("chunk_index")
        key = (filename, page, chunk_index)
        if key not in seen:
            seen.add(key)
            sources.append(
                SourceDocument(
                    filename=filename,
                    page=page,
                    chunk_index=chunk_index,
                    excerpt=doc.page_content[:200].strip(),
                )
            )
    return sources


# ── Main RAG executor ──────────────────────────────────────────────────────────


class RAGChain:
    """Encapsulates the full RAG pipeline for FinGuard AI."""

    def __init__(self) -> None:
        settings = get_settings()

        self._llm = ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            google_api_key=settings.google_api_key,
            temperature=settings.gemini_temperature,
            max_output_tokens=settings.gemini_max_output_tokens,
            convert_system_message_to_human=True,  # Gemini requirement
        )
        self._model_name = settings.gemini_model
        logger.info(f"RAGChain initialised with model '{self._model_name}'")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def run(self, question: str) -> AskResponse:
        """
        Execute the full RAG pipeline for *question*.

        Raises
        ------
        NoDocumentsIndexedError
            If the ChromaDB collection is empty.
        GeminiServiceError
            On Gemini API failures after retries.
        """
        vector_store = get_vector_store()

        if vector_store.collection_count() == 0:
            raise NoDocumentsIndexedError()

        # ── 1. Retrieve relevant chunks ──────────────────────────────────
        retriever = vector_store.as_retriever()
        logger.info(f"Retrieving context for: '{question[:80]}…'")

        try:
            retrieved_docs: list[Document] = retriever.invoke(question)
        except Exception as exc:
            raise GeminiServiceError(f"Retrieval failed: {exc}") from exc

        logger.info(f"  → {len(retrieved_docs)} chunk(s) retrieved")

        # ── 2. Build & run the LLM chain ────────────────────────────────
        chain = (
            {
                "context": lambda _: _format_docs(retrieved_docs),
                "question": RunnablePassthrough(),
            }
            | _PROMPT
            | self._llm
            | StrOutputParser()
        )

        try:
            answer: str = chain.invoke(question)
        except Exception as exc:
            raise GeminiServiceError(str(exc)) from exc

        # ── 3. Assemble response ─────────────────────────────────────────
        sources = _build_source_documents(retrieved_docs)
        logger.info(f"  → Answer generated, {len(sources)} unique source(s)")

        return AskResponse(
            question=question,
            answer=answer.strip(),
            sources=sources,
            model_used=self._model_name,
        )
