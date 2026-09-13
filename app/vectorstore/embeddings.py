"""Embedding model provider.

Wraps a HuggingFace sentence-transformers model behind LangChain's
`Embeddings` interface. The same model instance embeds both ingested
chunks (at index-build time) and incoming queries (at retrieval time),
which is what makes their vectors comparable in the first place.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from app.core.config import get_settings

if TYPE_CHECKING:
    from langchain_huggingface import HuggingFaceEmbeddings


@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    """Return the process-wide embedding model instance.

    The underlying model is downloaded from the Hugging Face Hub on first
    use and cached locally on disk afterward, so only the very first call
    per machine pays that cost. Cached in-process via `lru_cache` on top of
    that, so repeated calls reuse one loaded model instance instead of
    re-reading it from disk.

    `langchain_huggingface` (and the sentence-transformers/torch stack it
    pulls in) is imported here rather than at module level, so importing
    this module - or anything that depends on it - doesn't require that
    stack to be installed unless this function actually runs. Call sites
    that don't need a real model (tests injecting a fake `Embeddings`,
    for instance) are unaffected either way.
    """

    from langchain_huggingface import HuggingFaceEmbeddings

    settings = get_settings()
    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs={"device": settings.embedding_device},
    )
