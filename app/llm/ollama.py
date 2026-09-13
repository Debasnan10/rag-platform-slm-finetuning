"""LangChain <-> Ollama integration.

Ollama runs SLMs (small language models, e.g. llama3.2:3b) locally and
exposes them over a simple HTTP API on `localhost:11434`. LangChain ships
a wrapper (`langchain_ollama.OllamaLLM`) that turns that HTTP API into a
standard LangChain `BaseLanguageModel`, so the rest of our code (prompt
templates, RAG chains) never has to know it's talking to Ollama
specifically.

Note: `langchain_community.llms.Ollama` is deprecated upstream in favor of
the dedicated `langchain-ollama` package (`OllamaLLM`) used here - same
integration, actively maintained package instead of the legacy shim.
"""

from __future__ import annotations

from functools import lru_cache

import httpx
from langchain_ollama import OllamaLLM

from app.core.config import Settings, get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)


class OllamaProvider:
    """Concrete `LLMProvider` (see `app/llm/base.py`) backed by a local Ollama server."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings

    def get_llm(self) -> OllamaLLM:
        """Build a LangChain `OllamaLLM` configured from Settings.

        This does NOT make a network call - it just constructs the
        client object. The actual HTTP request happens the first time
        something calls `.invoke(...)` on it.
        """

        return OllamaLLM(
            base_url=self._settings.ollama_base_url,
            model=self._settings.ollama_rag_model,
            temperature=self._settings.ollama_temperature,
            timeout=self._settings.ollama_request_timeout_seconds,
        )

    def health_check(self) -> bool:
        """Ping Ollama's REST API to confirm the server is up and reachable.

        Hits `GET /api/tags` (Ollama's "list installed models" endpoint) -
        cheap, doesn't load a model into memory, just confirms the daemon
        is listening. Returns False (never raises) so callers - like the
        `/health` endpoint - can report degraded status instead of crashing.
        """

        url = f"{self._settings.ollama_base_url}/api/tags"
        try:
            response = httpx.get(url, timeout=5.0)
            response.raise_for_status()
            return True
        except httpx.HTTPError as exc:
            logger.warning("Ollama health check failed: %s", exc)
            return False

    def ensure_ready(self) -> None:
        """Raise `LLMError` if Ollama is not reachable.

        Use this at the point you're about to actually run a query -
        distinct from `health_check()`, which is used for status reporting
        and deliberately never raises.
        """

        if not self.health_check():
            raise LLMError(
                f"Cannot reach Ollama at {self._settings.ollama_base_url}. "
                "Is `ollama serve` running?"
            )


@lru_cache
def get_ollama_provider() -> OllamaProvider:
    """Process-wide singleton, same caching pattern as `get_settings()`."""

    return OllamaProvider(get_settings())
