"""
app/main.py
────────────
FinGuard AI — FastAPI application factory.

Startup sequence
----------------
1. Load & validate settings (crashes fast if GOOGLE_API_KEY is missing)
2. Ensure required directories exist
3. Pre-warm the VectorStore singleton (connects to ChromaDB)
4. Register exception handlers
5. Mount routers
"""

import sys
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from app.core.config import get_settings
from app.core.exceptions import FinGuardBaseError
from app.rag.vector_store import get_vector_store
from app.routes import ask, health, upload


# ── Logging ───────────────────────────────────────────────────────────────────

logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> — {message}",
    level="DEBUG",
)


# ── Lifespan (startup / shutdown) ─────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run startup logic before accepting requests."""
    logger.info("═══════════════════════════════════════════")
    logger.info("  FinGuard AI — starting up")
    logger.info("═══════════════════════════════════════════")

    settings = get_settings()
    settings.ensure_directories()
    logger.info(f"Upload directory  : {settings.upload_dir}")
    logger.info(f"ChromaDB directory: {settings.chroma_persist_dir}")

    # Pre-warm singletons so the first request is not slow
    get_vector_store()

    logger.info("Startup complete — ready to serve requests.")
    yield
    logger.info("FinGuard AI — shutting down.")


# ── Application factory ────────────────────────────────────────────────────────


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI-Powered Financial Compliance & Fraud Investigation Assistant. "
            "Upload regulatory PDFs, then ask natural-language questions — "
            "answers are grounded in your indexed documents."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],      # tighten in production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Exception handlers ────────────────────────────────────────────
    @app.exception_handler(FinGuardBaseError)
    async def finguard_exception_handler(
        request: Request, exc: FinGuardBaseError
    ) -> JSONResponse:
        logger.warning(f"FinGuardBaseError [{exc.status_code}]: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": type(exc).__name__,
                "detail": exc.message,
                "status_code": exc.status_code,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "InternalServerError",
                "detail": "An unexpected error occurred. Please try again.",
                "status_code": 500,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

    # ── Routers ───────────────────────────────────────────────────────
    app.include_router(health.router)
    app.include_router(upload.router)
    app.include_router(ask.router)

    return app


app = create_app()
