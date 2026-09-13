"""FastAPI application entrypoint.

`create_app()` is an application factory - a function that builds and
returns a configured `FastAPI` instance, rather than a bare module-level
`app = FastAPI()`. This gives tests a clean way to spin up a fresh app
instance (e.g. with different settings) instead of importing one shared
global.

Run locally with:
    uvicorn app.api.main:app --reload
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.core.config import Settings, get_settings
from app.core.exceptions import AppError
from app.core.logging import CorrelationIdMiddleware, get_logger, setup_logging
from app.llm.ollama import get_ollama_provider

logger = get_logger(__name__)


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup/shutdown hook: runs once before the app starts serving
    requests, and once as it shuts down.
    """

    settings = get_settings()
    setup_logging(level=settings.log_level, json_output=settings.log_json)
    logger.info("Starting %s (env=%s)", settings.app_name, settings.app_env)
    yield
    logger.info("Shutting down %s", settings.app_name)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured FastAPI app instance."""

    settings = settings or get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="Production-grade RAG platform with on-device SLM fine-tuning.",
        lifespan=_lifespan,
    )

    app.add_middleware(CorrelationIdMiddleware)

    @app.exception_handler(AppError)
    async def _handle_app_error(_: Request, exc: AppError) -> JSONResponse:
        """Translate any `AppError` subclass into a consistent JSON error body.

        One place that maps domain exceptions to HTTP responses, instead of
        try/except in every route.
        """

        logger.warning("%s: %s", type(exc).__name__, exc.message)
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": type(exc).__name__,
                "message": exc.message,
                "details": exc.details,
            },
        )

    @app.get("/health", tags=["system"])
    def health() -> dict:
        """Liveness/readiness probe.

        Reports the app as up, plus whether it can currently reach
        Ollama - useful for `docker compose` healthchecks and for
        catching "the API is fine but the LLM backend isn't" at a glance.
        """

        ollama_ok = get_ollama_provider().health_check()
        return {
            "status": "ok" if ollama_ok else "degraded",
            "app_name": settings.app_name,
            "app_version": settings.app_version,
            "environment": settings.app_env,
            "dependencies": {
                "ollama": "ok" if ollama_ok else "unreachable",
            },
        }

    return app


app = create_app()
