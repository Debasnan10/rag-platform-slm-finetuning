"""FAISS-backed vector store for document chunks.

Wraps LangChain's FAISS integration behind a small class that owns the
concerns specific to this app: which embedding model to use, where the
index lives on disk, and translating whatever FAISS/LangChain raise into
`VectorStoreError` / `NotFoundError` so callers only need to know about
this app's own exception hierarchy.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.retrievers import BaseRetriever

from app.core.config import Settings, get_settings
from app.core.exceptions import NotFoundError, VectorStoreError
from app.core.logging import get_logger
from app.vectorstore.embeddings import get_embeddings

logger = get_logger(__name__)


class VectorStore:
    """A single FAISS index over document chunks, with disk persistence.

    Holds at most one index at a time. `embeddings` is injectable
    (defaults to `get_embeddings()`) so tests can exercise the index
    build/search/persistence logic with a lightweight fake instead of
    loading a real sentence-transformers model.
    """

    def __init__(self, settings: Settings | None = None, embeddings: Embeddings | None = None) -> None:
        self._settings = settings or get_settings()
        self._embeddings = embeddings or get_embeddings()
        self._index: FAISS | None = None

    @property
    def is_ready(self) -> bool:
        """Whether an index is currently built or loaded and can be searched."""

        return self._index is not None

    def build(self, documents: list[Document]) -> None:
        """Build a fresh in-memory index from `documents`, discarding any existing one."""

        if not documents:
            raise VectorStoreError("Cannot build an index from zero documents.")

        try:
            self._index = FAISS.from_documents(documents, self._embeddings)
        except Exception as exc:  # noqa: BLE001 - normalize every backend's own exception type
            raise VectorStoreError(f"Failed to build FAISS index: {exc}") from exc

        logger.info("Built FAISS index from %d chunk(s)", len(documents))

    def add(self, documents: list[Document]) -> None:
        """Add more chunks to the current index, building one first if none exists yet."""

        if not documents:
            return

        if self._index is None:
            self.build(documents)
            return

        try:
            self._index.add_documents(documents)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to add documents to FAISS index: {exc}") from exc

        logger.info("Added %d chunk(s) to existing FAISS index", len(documents))

    def save(self, directory: str | Path | None = None) -> None:
        """Persist the current index to `directory` (default: `settings.faiss_index_dir`)."""

        if self._index is None:
            raise VectorStoreError("No index to save - call build() first.")

        target = Path(directory or self._settings.faiss_index_dir)
        target.mkdir(parents=True, exist_ok=True)
        self._index.save_local(str(target))
        logger.info("Saved FAISS index to %s", target)

    def load(self, directory: str | Path | None = None) -> None:
        """Load a previously-saved index from `directory` (default: `settings.faiss_index_dir`)."""

        target = Path(directory or self._settings.faiss_index_dir)
        if not target.exists():
            raise NotFoundError(f"No FAISS index found at {target}", details={"path": str(target)})

        try:
            # FAISS persists its docstore via pickle alongside the raw index
            # bytes, so LangChain requires this flag to load it back.
            # Acceptable here because the only indexes this app ever loads are
            # ones it wrote itself to `faiss_index_dir`.
            self._index = FAISS.load_local(
                str(target), self._embeddings, allow_dangerous_deserialization=True
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"Failed to load FAISS index from {target}: {exc}") from exc

        logger.info("Loaded FAISS index from %s", target)

    def similarity_search(self, query: str, k: int | None = None) -> list[Document]:
        """Return the `k` chunks most similar to `query` (default: `settings.faiss_top_k`)."""

        if self._index is None:
            raise VectorStoreError("No index built or loaded - call build() or load() first.")

        return self._index.similarity_search(query, k=k or self._settings.faiss_top_k)

    def similarity_search_with_score(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """Like `similarity_search`, but pairs each chunk with its L2 distance to `query`."""

        if self._index is None:
            raise VectorStoreError("No index built or loaded - call build() or load() first.")

        return self._index.similarity_search_with_score(query, k=k or self._settings.faiss_top_k)

    def as_retriever(self, k: int | None = None) -> BaseRetriever:
        """Expose this index as a LangChain `BaseRetriever` (for use in `EnsembleRetriever`, etc.)."""

        if self._index is None:
            raise VectorStoreError("No index built or loaded - call build() or load() first.")

        return self._index.as_retriever(search_kwargs={"k": k or self._settings.faiss_top_k})


@lru_cache
def get_vector_store() -> VectorStore:
    """Return the process-wide `VectorStore` instance."""

    return VectorStore()
