"""Tests for hybrid retrieval (dense + sparse) and cross-encoder reranking."""

from __future__ import annotations

import hashlib
import re

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.exceptions import RetrievalError
from app.retrieval.hybrid import build_hybrid_retriever
from app.retrieval.reranker import rerank
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
def built_vector_store() -> VectorStore:
    store = VectorStore(embeddings=_FakeEmbeddings())
    store.build(DOCUMENTS)
    return store


def test_hybrid_retriever_finds_the_relevant_chunk(built_vector_store: VectorStore) -> None:
    retriever = build_hybrid_retriever(DOCUMENTS, vector_store=built_vector_store)

    results = retriever.invoke("interest rates inflation bank")

    assert any(result.metadata["topic"] == "finance" for result in results)


def test_hybrid_retriever_rejects_empty_documents(built_vector_store: VectorStore) -> None:
    with pytest.raises(RetrievalError):
        build_hybrid_retriever([], vector_store=built_vector_store)


def test_hybrid_retriever_rejects_an_unready_vector_store() -> None:
    empty_store = VectorStore(embeddings=_FakeEmbeddings())

    with pytest.raises(RetrievalError):
        build_hybrid_retriever(DOCUMENTS, vector_store=empty_store)


def _keyword_overlap_score(query: str, passage: str) -> float:
    """Deterministic fake scorer: fraction of query words present in the passage."""

    query_words = set(re.findall(r"[a-z]+", query.lower()))
    passage_words = set(re.findall(r"[a-z]+", passage.lower()))
    if not query_words:
        return 0.0
    return len(query_words & passage_words) / len(query_words)


def test_rerank_orders_by_relevance() -> None:
    results = rerank(
        "interest rates inflation",
        [DOC_ASTRONOMY, DOC_FINANCE],
        top_n=2,
        score_fn=_keyword_overlap_score,
    )

    assert results[0].metadata["topic"] == "finance"
    assert results[1].metadata["topic"] == "astronomy"


def test_rerank_truncates_to_top_n() -> None:
    results = rerank(
        "interest rates inflation",
        [DOC_ASTRONOMY, DOC_FINANCE],
        top_n=1,
        score_fn=_keyword_overlap_score,
    )

    assert len(results) == 1
    assert results[0].metadata["topic"] == "finance"


def test_rerank_with_no_documents_returns_empty_list() -> None:
    assert rerank("anything", [], score_fn=_keyword_overlap_score) == []
