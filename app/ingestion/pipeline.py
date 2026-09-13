"""Ingestion pipeline: load -> split -> attach metadata.

This module doesn't introduce any new logic of its own - it calls
`loader.load_document()` then `splitter.split_documents()` in sequence and
tags every resulting chunk with where it came from. Keeping this
orchestration separate from `loader.py`/`splitter.py` keeps each of those
two independently testable, while this file is the one place that
downstream consumers (vector store ingestion, the `/ingest` API route)
call into.
"""

from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.ingestion.loader import load_document
from app.ingestion.splitter import split_documents

logger = get_logger(__name__)


def ingest_file(file_path: str | Path, settings: Settings | None = None) -> list[Document]:
    """Load one file and return its chunks, tagged with source metadata."""

    settings = settings or get_settings()
    path = Path(file_path)

    documents = load_document(path)
    chunks = split_documents(documents, settings=settings)

    for index, chunk in enumerate(chunks):
        chunk.metadata["source_file"] = path.name
        chunk.metadata["chunk_index"] = index

    logger.info("Ingested %s -> %d chunk(s)", path.name, len(chunks))
    return chunks


def ingest_directory(directory: str | Path, settings: Settings | None = None) -> list[Document]:
    """Ingest every supported file directly inside `directory` (non-recursive).

    Uses `settings.ingestion_supported_extensions_list` (see
    `app/core/config.py`) to decide which files to pick up, so the
    supported extensions stay configured in exactly one place.
    """

    settings = settings or get_settings()
    directory = Path(directory)
    allowed_extensions = set(settings.ingestion_supported_extensions_list)

    all_chunks: list[Document] = []
    for file_path in sorted(directory.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() in allowed_extensions:
            all_chunks.extend(ingest_file(file_path, settings=settings))

    logger.info("Ingested directory %s -> %d total chunk(s)", directory, len(all_chunks))
    return all_chunks
