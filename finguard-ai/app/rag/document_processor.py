"""
app/rag/document_processor.py
──────────────────────────────
Responsible for:
  1. Loading a PDF with PyPDFLoader
  2. Splitting it into chunks via RecursiveCharacterTextSplitter
  3. Attaching metadata (filename, page, chunk_index)
  4. Returning the list of LangChain Document objects

This module is intentionally pure — it has no ChromaDB or embedding
dependencies so it can be tested in isolation.
"""

from pathlib import Path

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from loguru import logger

from app.core.config import get_settings
from app.core.exceptions import DocumentNotFoundError, EmptyPDFError


def load_and_chunk_pdf(file_path: Path) -> list[Document]:
    """
    Load a PDF from *file_path*, split into overlapping chunks, and
    attach rich metadata to every chunk.

    Returns
    -------
    list[Document]
        Non-empty list of chunked LangChain Document objects.

    Raises
    ------
    DocumentNotFoundError
        If the file does not exist on disk.
    EmptyPDFError
        If the PDF contains no extractable text.
    """
    settings = get_settings()

    if not file_path.exists():
        raise DocumentNotFoundError(file_path.name)

    logger.info(f"Loading PDF: {file_path.name}")

    # ── 1. Load pages ─────────────────────────────────────────────────
    loader = PyPDFLoader(str(file_path))
    pages: list[Document] = loader.load()

    if not pages or all(not p.page_content.strip() for p in pages):
        raise EmptyPDFError(file_path.name)

    raw_text_len = sum(len(p.page_content) for p in pages)
    logger.info(f"  → {len(pages)} page(s), {raw_text_len:,} raw characters")

    # ── 2. Chunk ──────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Document] = splitter.split_documents(pages)

    if not chunks:
        raise EmptyPDFError(file_path.name)

    # ── 3. Enrich metadata ────────────────────────────────────────────
    for idx, chunk in enumerate(chunks):
        chunk.metadata.update(
            {
                "filename": file_path.name,
                "chunk_index": idx,
                "total_chunks": len(chunks),
                # PyPDFLoader stores page number under "page" (0-based int)
                "page": chunk.metadata.get("page", 0) + 1,  # make 1-based
                "source": file_path.name,  # LangChain convention
            }
        )

    logger.info(
        f"  → {len(chunks)} chunk(s) created "
        f"(size={settings.chunk_size}, overlap={settings.chunk_overlap})"
    )
    return chunks
