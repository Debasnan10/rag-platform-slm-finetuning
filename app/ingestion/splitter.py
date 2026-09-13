"""Text splitting via LangChain's RecursiveCharacterTextSplitter.

Large language models (and embedding models) can only handle a limited
amount of text at once, so a loaded document has to be cut into smaller
"chunks" before it can be embedded and stored. `RecursiveCharacterTextSplitter`
does this by trying a list of separators in priority order - paragraph
breaks first, then sentences, then words, then raw characters as a last
resort - which tends to keep chunks at natural boundaries rather than
cutting a sentence in half. We configure it, but the splitting logic itself
is LangChain's, not ours.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import Settings, get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def split_documents(documents: list[Document], settings: Settings | None = None) -> list[Document]:
    """Split loaded documents into `settings.chunk_size`-ish pieces.

    `chunk_overlap` characters are repeated between consecutive chunks so
    that a sentence or idea spanning a chunk boundary isn't lost entirely
    from either side.
    """

    settings = settings or get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    chunks = splitter.split_documents(documents)

    logger.info(
        "Split %d document(s) into %d chunk(s) (chunk_size=%d, chunk_overlap=%d)",
        len(documents),
        len(chunks),
        settings.chunk_size,
        settings.chunk_overlap,
    )
    return chunks
