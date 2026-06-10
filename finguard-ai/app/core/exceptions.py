"""
app/core/exceptions.py
──────────────────────
Domain-specific exceptions for FinGuard AI.
FastAPI exception handlers are registered in main.py.
"""


class FinGuardBaseError(Exception):
    """Root exception for all FinGuard errors."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        self.message = message
        self.status_code = status_code
        super().__init__(message)


# ── Upload / document errors ──────────────────────────────────────────


class InvalidFileTypeError(FinGuardBaseError):
    def __init__(self, filename: str) -> None:
        super().__init__(
            message=f"'{filename}' is not a supported file type. Only PDF files are accepted.",
            status_code=415,
        )


class FileTooLargeError(FinGuardBaseError):
    def __init__(self, max_mb: int) -> None:
        super().__init__(
            message=f"Uploaded file exceeds the maximum allowed size of {max_mb} MB.",
            status_code=413,
        )


class EmptyPDFError(FinGuardBaseError):
    def __init__(self, filename: str) -> None:
        super().__init__(
            message=f"'{filename}' contains no extractable text. The PDF may be scanned or image-only.",
            status_code=422,
        )


class DocumentNotFoundError(FinGuardBaseError):
    def __init__(self, filename: str) -> None:
        super().__init__(
            message=f"Document '{filename}' was not found on the server.",
            status_code=404,
        )


# ── RAG / query errors ────────────────────────────────────────────────


class EmptyQuestionError(FinGuardBaseError):
    def __init__(self) -> None:
        super().__init__(
            message="Question must not be empty.",
            status_code=400,
        )


class NoDocumentsIndexedError(FinGuardBaseError):
    def __init__(self) -> None:
        super().__init__(
            message="No documents have been indexed yet. Please upload at least one PDF before asking questions.",
            status_code=409,
        )


# ── External service errors ───────────────────────────────────────────


class GeminiServiceError(FinGuardBaseError):
    def __init__(self, detail: str = "") -> None:
        msg = "The Gemini AI service returned an error."
        if detail:
            msg += f" Detail: {detail}"
        super().__init__(message=msg, status_code=502)


class ChromaServiceError(FinGuardBaseError):
    def __init__(self, detail: str = "") -> None:
        msg = "The ChromaDB vector store encountered an error."
        if detail:
            msg += f" Detail: {detail}"
        super().__init__(message=msg, status_code=502)
