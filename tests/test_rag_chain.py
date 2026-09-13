"""Tests for the end-to-end RAG chain (retrieve -> rerank -> prompt -> generate)."""

from __future__ import annotations

import hashlib
import re

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseLanguageModel

from app.rag.chain import RAGChain
from app.vectorstore.store import VectorStore


class _FakeEmbeddings(Embeddings):
    """See `tests/test_vectorstore.py` for why this exists instead of a real model."""

    _DIMENSIONS = 256

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._DIMENSIONS
        for word in re.findall(r"[a-z]+", text.lower()):
            bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self._DIMENSIONS
            vector[bucket] += 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class _FakeLLM:
    """Echoes back the prompt it was asked to answer with, so tests can assert
    that retrieved context actually reached the model - without running a
    real Ollama server."""

    last_prompt: str | None = None

    def invoke(self, prompt: str) -> str:
        _FakeLLM.last_prompt = prompt
        return "The stubbed answer."


class _FakeLLMProvider:
    """Implements `app.llm.base.LLMProvider` structurally, for tests."""

    def get_llm(self) -> BaseLanguageModel:
        return _FakeLLM()  # type: ignore[return-value]

    def health_check(self) -> bool:
        return True


def _keyword_overlap_score(query: str, passage: str) -> float:
    """Deterministic fake reranker scorer: fraction of query words in the passage."""

    query_words = set(re.findall(r"[a-z]+", query.lower()))
    passage_words = set(re.findall(r"[a-z]+", passage.lower()))
    if not query_words:
        return 0.0
    return len(query_words & passage_words) / len(query_words)


DOC_ASTRONOMY = Document(
    page_content="Jupiter is the largest planet in the solar system and has dozens of moons.",
    metadata={"topic": "astronomy"},
)
DOC_FINANCE = Document(
    page_content="The central bank raised interest rates to curb rising inflation.",
    metadata={"topic": "finance"},
)
DOCUMENTS = [DOC_ASTRONOMY, DOC_FINANCE]


@pytest.fixture()
def chain() -> RAGChain:
    store = VectorStore(embeddings=_FakeEmbeddings())
    store.build(DOCUMENTS)
    return RAGChain(
        DOCUMENTS,
        vector_store=store,
        llm_provider=_FakeLLMProvider(),
        score_fn=_keyword_overlap_score,
    )


def test_answer_grounds_the_prompt_in_retrieved_context(chain: RAGChain) -> None:
    result = chain.answer("What happened to interest rates?")

    assert result.answer == "The stubbed answer."
    assert result.sources[0].metadata["topic"] == "finance"
    assert "central bank" in _FakeLLM.last_prompt
    assert "What happened to interest rates?" in _FakeLLM.last_prompt


def test_answer_returns_sources_alongside_the_answer(chain: RAGChain) -> None:
    result = chain.answer("planet moons solar system")

    assert result.sources
    assert all(isinstance(source, Document) for source in result.sources)
