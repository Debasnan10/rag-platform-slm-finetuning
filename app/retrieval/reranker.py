"""Cross-encoder reranking of retrieved chunks.

A bi-encoder (the embedding model in `app/vectorstore/embeddings.py`)
scores query and passage independently, which is what makes a vector
index fast to search but caps how precisely it can judge relevance. A
cross-encoder scores the (query, passage) pair jointly - slower, so it
only runs over the small candidate set retrieval already narrowed down to,
not the whole corpus - and reorders that set by actual relevance.
"""

from __future__ import annotations

from collections.abc import Callable
from functools import lru_cache

from langchain_core.documents import Document

from app.core.config import get_settings

#: (query, passage) -> relevance score, higher is more relevant.
ScoreFn = Callable[[str, str], float]


def rerank(
    query: str,
    documents: list[Document],
    top_n: int | None = None,
    score_fn: ScoreFn | None = None,
) -> list[Document]:
    """Return the `top_n` of `documents` most relevant to `query`.

    `score_fn` is injectable (defaults to a real cross-encoder model) so
    tests can supply a cheap deterministic scorer instead of loading real
    model weights.
    """

    if not documents:
        return []

    settings = get_settings()
    top_n = top_n or settings.reranker_top_n
    scorer = score_fn or _cross_encoder_score

    scored = [(document, scorer(query, document.page_content)) for document in documents]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return [document for document, _ in scored[:top_n]]


def _cross_encoder_score(query: str, passage: str) -> float:
    model = _get_cross_encoder()
    return float(model.predict([(query, passage)])[0])


@lru_cache
def _get_cross_encoder():
    # Deferred: sentence-transformers pulls in torch, so this import only
    # runs (and only needs to be installed) when reranking actually happens
    # with the real model, not merely when this module is imported.
    from sentence_transformers import CrossEncoder

    settings = get_settings()
    return CrossEncoder(settings.reranker_model_name)
