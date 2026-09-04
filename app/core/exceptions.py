"""Custom exception hierarchy for the RAG platform.

Every exception the application deliberately raises inherits from
`AppError`. This mirrors a typical C# pattern of a base custom
`ApplicationException` with specific subclasses (e.g.
`ValidationException : ApplicationException`) - it lets calling code
either catch broad (`except AppError`) or narrow (`except VectorStoreError`),
and lets a single FastAPI exception handler translate *any* `AppError`
into a well-formed HTTP response without knowing about every subclass.
"""

from __future__ import annotations


class AppError(Exception):
    """Base class for all application-raised (expected) errors.

    Attributes:
        message: Human-readable explanation, safe to show to a caller.
        details: Optional structured context (e.g. {"file": "a.pdf"})
            useful in logs but not necessarily shown to end users.
    """

    #: Default HTTP status code used when this error reaches the API layer.
    status_code: int = 500

    def __init__(self, message: str, *, details: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConfigurationError(AppError):
    """Raised when required configuration is missing or invalid."""

    status_code = 500


class IngestionError(AppError):
    """Raised when a document fails to load, parse, or split (Phase 2)."""

    status_code = 422


class VectorStoreError(AppError):
    """Raised for FAISS index build/read/persist failures (Phase 3)."""

    status_code = 500


class RetrievalError(AppError):
    """Raised when dense/sparse/hybrid retrieval fails (Phase 4)."""

    status_code = 500


class LLMError(AppError):
    """Raised when the LLM backend (Ollama) is unreachable or errors out."""

    status_code = 502


class FineTuningError(AppError):
    """Raised for LoRA/QLoRA training or adapter loading failures (Phase 6-7)."""

    status_code = 500


class EvaluationError(AppError):
    """Raised when metric computation or evaluation runs fail (Phase 8)."""

    status_code = 500


class NotFoundError(AppError):
    """Raised when a requested resource (document, index, adapter) doesn't exist."""

    status_code = 404
