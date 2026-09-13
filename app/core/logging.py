"""Structured logging with request-scoped correlation IDs.

Every log record is emitted as one JSON object (when `log_json=True`),
carrying a `correlation_id` that is constant for the lifetime of a single
request - so filtering logs by that ID reconstructs a request's full
execution trace across modules.

The correlation ID is held in a `contextvars.ContextVar` rather than a
global or a parameter threaded through every call. `ContextVar` values are
isolated per asyncio task, so concurrent requests handled on the same
event loop never observe each other's value.
"""

from __future__ import annotations

import json
import logging
import sys
import time
import uuid
from contextvars import ContextVar
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

# The ContextVar holding the "current" correlation ID. Default "-" means
# "no request in flight" (e.g. a background job or a unit test that never
# set one).
_correlation_id: ContextVar[str] = ContextVar("correlation_id", default="-")

CORRELATION_ID_HEADER = "X-Correlation-ID"


def get_correlation_id() -> str:
    """Return the correlation ID for the current request/task, if any."""

    return _correlation_id.get()


def set_correlation_id(correlation_id: str | None = None) -> str:
    """Set (or generate) the correlation ID for the current context."""

    value = correlation_id or str(uuid.uuid4())
    _correlation_id.set(value)
    return value


class _CorrelationIdFilter(logging.Filter):
    """Injects the current correlation ID into every LogRecord."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.correlation_id = get_correlation_id()
        return True


class _JsonFormatter(logging.Formatter):
    """Renders each LogRecord as a single JSON line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": getattr(record, "correlation_id", "-"),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


class _PlainFormatter(logging.Formatter):
    """Human-readable formatter, handy for local development in a terminal."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s | %(levelname)-8s | %(correlation_id)s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )


def setup_logging(*, level: str = "INFO", json_output: bool = True) -> None:
    """Configure the root logger. Call once, at process startup.

    Replaces any handlers already attached to the root logger, so calling
    this more than once (e.g. across repeated test-suite app instances)
    does not produce duplicated log output.
    """

    root = logging.getLogger()
    root.setLevel(level)

    # Remove any handlers a previous setup_logging() call (or a library)
    # already attached, so we don't end up with duplicated log lines.
    for existing in list(root.handlers):
        root.removeHandler(existing)

    handler = logging.StreamHandler(stream=sys.stdout)
    handler.addFilter(_CorrelationIdFilter())
    handler.setFormatter(_JsonFormatter() if json_output else _PlainFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a module-scoped logger, e.g. `get_logger(__name__)`."""

    return logging.getLogger(name)


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """FastAPI/Starlette middleware that assigns a correlation ID per request.

    - Reuses the caller's `X-Correlation-ID` header if they sent one
      (useful when this API is called by another service that already has
      a trace ID), otherwise generates a fresh UUID4.
    - Echoes the ID back on the response so the caller can log it too.
    - Logs one line per request with method, path, status, and latency.
    """

    async def dispatch(self, request: Request, call_next):  # type: ignore[override]
        incoming = request.headers.get(CORRELATION_ID_HEADER)
        correlation_id = set_correlation_id(incoming)
        logger = get_logger("app.request")

        start = time.perf_counter()
        response: Response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000

        response.headers[CORRELATION_ID_HEADER] = correlation_id
        logger.info(
            "%s %s -> %s (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
