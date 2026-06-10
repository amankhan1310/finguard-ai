"""
app/core/config.py
──────────────────
Centralised settings loaded from environment variables / .env file.
All other modules import from here — never from os.environ directly.
"""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── App ────────────────────────────────────────────────────────────
    app_name: str = "FinGuard AI"
    app_version: str = "1.0.0"
    debug: bool = False

    # ── Google / Gemini ────────────────────────────────────────────────
    google_api_key: str

    gemini_model: str = "gemini-1.5-flash"
    gemini_temperature: float = 0.2
    gemini_max_output_tokens: int = 2048

    embedding_model: str = "models/text-embedding-004"

    # ── Storage ────────────────────────────────────────────────────────
    upload_dir: Path = Path("data/uploads")
    max_upload_size_mb: int = 50

    # ── ChromaDB ───────────────────────────────────────────────────────
    chroma_persist_dir: str = "chroma_db"
    chroma_collection_name: str = "finguard_documents"

    # ── RAG ────────────────────────────────────────────────────────────
    chunk_size: int = 1000
    chunk_overlap: int = 200
    retriever_top_k: int = 5

    # ── Derived helpers ────────────────────────────────────────────────
    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024

    def ensure_directories(self) -> None:
        """Create required directories if they don't already exist."""
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
