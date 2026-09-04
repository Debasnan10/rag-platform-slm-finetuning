"""Phase 1 smoke tests.

These don't test business logic (there isn't any yet) - they confirm the
skeleton itself is sound: config loads, logging/correlation IDs work, the
exception hierarchy behaves, and the FastAPI app boots and serves /health.
Nothing here requires Ollama, Docker, or the network to be running -
except the /health test, which is written to pass either way (it reports
"degraded" rather than failing if Ollama happens to be offline).
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.core.exceptions import AppError, LLMError, VectorStoreError
from app.core.logging import get_correlation_id, set_correlation_id


def test_settings_load_with_sane_defaults() -> None:
    settings = Settings()

    assert settings.app_name == "RAG Platform"
    assert settings.chunk_size == 500
    assert settings.chunk_overlap == 50
    assert settings.ollama_rag_model == "llama3.2:3b"


def test_get_settings_is_cached_singleton() -> None:
    assert get_settings() is get_settings()


def test_ingestion_extensions_are_parsed_into_a_list() -> None:
    settings = Settings(ingestion_supported_extensions=".pdf,.docx , .md")

    assert settings.ingestion_supported_extensions_list == [".pdf", ".docx", ".md"]


def test_correlation_id_defaults_and_can_be_set() -> None:
    assert get_correlation_id() == "-"

    generated = set_correlation_id()
    assert get_correlation_id() == generated

    set_correlation_id("fixed-id-123")
    assert get_correlation_id() == "fixed-id-123"


def test_exception_hierarchy() -> None:
    assert issubclass(LLMError, AppError)
    assert issubclass(VectorStoreError, AppError)

    err = LLMError("Ollama unreachable", details={"url": "http://localhost:11434"})
    assert err.status_code == 502
    assert err.message == "Ollama unreachable"
    assert err.details == {"url": "http://localhost:11434"}


def test_health_endpoint_returns_200_and_expected_shape(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] in {"ok", "degraded"}
    assert body["app_name"] == "RAG Platform"
    assert "ollama" in body["dependencies"]


def test_health_endpoint_echoes_correlation_id_header(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Correlation-ID": "test-abc"})

    assert response.headers["X-Correlation-ID"] == "test-abc"


def test_unset_correlation_id_header_generates_one(client: TestClient) -> None:
    response = client.get("/health")

    assert response.headers.get("X-Correlation-ID")
