"""Tests for the FAISS vector store: build, persistence, and search."""

from __future__ import annotations

import hashlib
import re

import pytest
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.core.exceptions import NotFoundError, VectorStoreError
from app.ingestion.pipeline import ingest_directory
from app.vectorstore.store import VectorStore, get_vector_store


class _FakeEmbeddings(Embeddings):
    """Deterministic, offline stand-in for a real embedding model.

    Feature-hashes each word into one of a fixed number of buckets and
    counts occurrences, so texts sharing vocabulary land close together in
    this space. That's enough to exercise FAISS build/search/persistence
    logic without downloading a real sentence-transformers model in tests.
    """

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


@pytest.fixture()
def store() -> VectorStore:
    return VectorStore(embeddings=_FakeEmbeddings())


def test_build_creates_a_searchable_index(store: VectorStore) -> None:
    store.build([DOC_ASTRONOMY, DOC_FINANCE])

    assert store.is_ready
    results = store.similarity_search("planet moons solar system", k=1)
    assert results[0].metadata["topic"] == "astronomy"


def test_build_with_no_documents_raises(store: VectorStore) -> None:
    with pytest.raises(VectorStoreError):
        store.build([])


def test_search_before_build_or_load_raises(store: VectorStore) -> None:
    with pytest.raises(VectorStoreError):
        store.similarity_search("anything")


def test_add_builds_index_when_none_exists(store: VectorStore) -> None:
    store.add([DOC_ASTRONOMY])

    assert store.is_ready
    assert store.similarity_search("planet")[0].metadata["topic"] == "astronomy"


def test_add_extends_an_existing_index(store: VectorStore) -> None:
    store.build([DOC_ASTRONOMY])
    store.add([DOC_FINANCE])

    results = store.similarity_search("interest rates inflation bank", k=1)
    assert results[0].metadata["topic"] == "finance"


def test_save_then_load_round_trip(store: VectorStore, tmp_path) -> None:
    store.build([DOC_ASTRONOMY, DOC_FINANCE])
    store.save(tmp_path)

    reloaded = VectorStore(embeddings=_FakeEmbeddings())
    reloaded.load(tmp_path)

    results = reloaded.similarity_search("planet moons solar system", k=1)
    assert results[0].metadata["topic"] == "astronomy"


def test_load_missing_directory_raises_not_found(store: VectorStore, tmp_path) -> None:
    with pytest.raises(NotFoundError):
        store.load(tmp_path / "does-not-exist")


def test_similarity_search_with_score_returns_distance(store: VectorStore) -> None:
    store.build([DOC_ASTRONOMY, DOC_FINANCE])

    results = store.similarity_search_with_score("planet moons solar system", k=2)

    assert len(results) == 2
    for document, distance in results:
        assert isinstance(document, Document)
        assert distance >= 0.0


def test_get_vector_store_returns_a_singleton(monkeypatch: pytest.MonkeyPatch) -> None:
    """`get_vector_store()` caches one instance per process (see `get_settings()`).

    The real embedding model is swapped out here so this test doesn't pay
    for (or depend on) a network fetch just to prove the caching contract.
    """

    monkeypatch.setattr("app.vectorstore.store.get_embeddings", lambda: _FakeEmbeddings())
    get_vector_store.cache_clear()
    try:
        assert get_vector_store() is get_vector_store()
    finally:
        get_vector_store.cache_clear()


def test_ingested_fixtures_are_indexed_and_searchable(tmp_path) -> None:
    """End-to-end: ingestion pipeline output feeds straight into the vector store."""

    chunks = ingest_directory("tests/fixtures")
    store = VectorStore(embeddings=_FakeEmbeddings())

    store.build(chunks)
    store.save(tmp_path)

    results = store.similarity_search("knee replacement recovery", k=3)
    assert len(results) == 3
    assert all(result.metadata.get("source_file") for result in results)
