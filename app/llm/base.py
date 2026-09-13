"""LangChain LLM abstraction.

The rest of the app (RAG chain, evaluation, fine-tuning comparisons)
depends on this thin `Protocol`, not on a concrete LangChain class
directly, keeping call sites swappable and easy to fake in tests.

Today there is exactly one implementation (`app/llm/ollama.py`, backed by
Ollama). Adding another local runtime (e.g. llama.cpp directly, or a
different model class) only requires satisfying this Protocol and wiring
it up at the provider's construction site.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from langchain_core.language_models import BaseLanguageModel


@runtime_checkable
class LLMProvider(Protocol):
    """Anything that can hand back a configured LangChain LLM."""

    def get_llm(self) -> BaseLanguageModel:
        """Return a ready-to-use LangChain LLM/ChatModel instance."""
        ...

    def health_check(self) -> bool:
        """Return True if the underlying LLM backend is reachable."""
        ...
