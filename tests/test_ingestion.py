"""Phase 2 tests: document loaders, the text splitter, and the pipeline.

All tests run against the real fixture files in tests/fixtures/ (a
fictional post-op care guideline, provided as .pdf, .docx, .html, .md) -
no mocking of the loaders themselves, since parsing real files correctly
is the whole point of this phase.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from app.core.config import Settings
from app.core.exceptions import IngestionError
from app.ingestion.loader import load_document, supported_extensions
from app.ingestion.pipeline import ingest_directory, ingest_file
from app.ingestion.splitter import split_documents

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize("filename", ["sample.pdf", "sample.docx", "sample.html", "sample.md"])
def test_load_document_extracts_expected_content(filename: str) -> None:
    documents = load_document(FIXTURES_DIR / filename)

    assert len(documents) >= 1
    full_text = " ".join(doc.page_content for doc in documents)
    assert "Post-Operative Care" in full_text
    assert "knee" in full_text.lower()


def test_load_document_missing_file_raises_ingestion_error() -> None:
    with pytest.raises(IngestionError):
        load_document(FIXTURES_DIR / "does_not_exist.pdf")


def test_load_document_unsupported_extension_raises_ingestion_error(tmp_path: Path) -> None:
    bad_file = tmp_path / "notes.txt"
    bad_file.write_text("some text")

    with pytest.raises(IngestionError):
        load_document(bad_file)


def test_supported_extensions_lists_all_four_formats() -> None:
    assert supported_extensions() == [".docx", ".html", ".md", ".pdf"]


def test_split_documents_respects_configured_chunk_size() -> None:
    documents = load_document(FIXTURES_DIR / "sample.md")
    settings = Settings(chunk_size=200, chunk_overlap=20)

    chunks = split_documents(documents, settings=settings)

    assert len(chunks) > 1
    # A little slack over chunk_size is expected: the splitter prefers
    # breaking on paragraph/sentence boundaries over cutting exactly at N.
    assert all(len(chunk.page_content) <= 260 for chunk in chunks)


def test_ingest_file_tags_every_chunk_with_source_metadata() -> None:
    chunks = ingest_file(FIXTURES_DIR / "sample.html")

    assert len(chunks) > 0
    for index, chunk in enumerate(chunks):
        assert chunk.metadata["source_file"] == "sample.html"
        assert chunk.metadata["chunk_index"] == index


def test_ingest_directory_picks_up_all_supported_fixtures() -> None:
    chunks = ingest_directory(FIXTURES_DIR)

    source_files = {chunk.metadata["source_file"] for chunk in chunks}
    assert source_files == {"sample.pdf", "sample.docx", "sample.html", "sample.md"}
