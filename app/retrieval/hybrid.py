"""Hybrid retrieval: dense (FAISS) and sparse (BM25) results combined.

Dense retrieval (embedding similarity) finds chunks that are semantically
related to a query even without shared vocabulary; sparse retrieval (BM25,
a keyword-frequency ranking algorithm) finds chunks that share exact terms
with the query, which dense retrieval can miss for specific names, codes,
or jargon. Combining both covers each one's blind spot.

LangChain's `EnsembleRetriever` does the combining: it runs every retriever
independently, then fuses the ranked lists via Reciprocal Rank Fusion,
weighted by `settings.retrieval_dense_weight` / `retrieval_sparse_weight`.
"""

from __future__ import annotations

from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.core.config import Settings, get_settings
from app.core.exceptions import RetrievalError
from app.vectorstore.store import VectorStore


def build_hybrid_retriever(
    documents: list[Document],
    vector_store: VectorStore,
    settings: Settings | None = None,
) -> BaseRetriever:
    """Build a dense+sparse ensemble retriever over `documents`.

    `documents` should be the same chunk set the vector store's index was
    built from - BM25 is rebuilt from them here since, unlike the FAISS
    index, it isn't persisted to disk between calls.
    """

    settings = settings or get_settings()

    if not documents:
        raise RetrievalError("Cannot build a retriever from zero documents.")
    if not vector_store.is_ready:
        raise RetrievalError("Vector store has no index built or loaded yet.")

    try:
        sparse_retriever = BM25Retriever.from_documents(documents)
    except Exception as exc:  # noqa: BLE001 - normalize every backend's own exception type
        raise RetrievalError(f"Failed to build BM25 retriever: {exc}") from exc

    sparse_retriever.k = settings.faiss_top_k
    dense_retriever = vector_store.as_retriever(k=settings.faiss_top_k)

    return EnsembleRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        weights=[settings.retrieval_dense_weight, settings.retrieval_sparse_weight],
    )
