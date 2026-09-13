"""Document loading via LangChain's built-in loaders.

One LangChain loader per supported file extension. Each loader knows how
to open its specific file format and hand back a list of LangChain
`Document` objects - plain text plus a `metadata` dict (at minimum, the
source file path). We don't parse PDFs/DOCX/HTML ourselves; that parsing
logic already lives in these libraries (pypdf, docx2txt, BeautifulSoup) and
LangChain wraps them behind one consistent `.load()` interface.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from langchain_community.document_loaders import (
    BSHTMLLoader,
    Docx2txtLoader,
    PyPDFLoader,
    TextLoader,
)
from langchain_community.document_loaders.base import BaseLoader
from langchain_core.documents import Document

from app.core.exceptions import IngestionError
from app.core.logging import get_logger

logger = get_logger(__name__)


def _build_html_loader(path: str) -> BaseLoader:
    # `features="html.parser"` pins BSHTMLLoader to Python's built-in HTML
    # parser instead of its default of `lxml` - which isn't in
    # requirements.txt. This is the same reasoning as choosing BSHTMLLoader
    # itself over UnstructuredHTMLLoader (see below): prefer the option
    # with the smallest dependency footprint that still gets the job done.
    return BSHTMLLoader(path, bs_kwargs={"features": "html.parser"})


# Maps a file extension to a factory that builds the loader for it. A
# factory (not just the bare class) so each extension can pass whatever
# constructor arguments it needs - every entry still produces something
# with the same shape: a `.load()` method returning `list[Document]`.
_LOADER_FACTORY_BY_EXTENSION: dict[str, Callable[[str], BaseLoader]] = {
    ".pdf": PyPDFLoader,
    ".docx": Docx2txtLoader,
    # BSHTMLLoader (BeautifulSoup-based) rather than the brief's suggested
    # UnstructuredHTMLLoader: the latter needs NLTK to download tokenizer
    # data from the internet on first use, which is fragile - it fails
    # outright in network-restricted environments and is a needless
    # runtime dependency for what is, structurally, just "strip the HTML
    # tags and keep the text."
    ".html": _build_html_loader,
    ".md": TextLoader,
}


def load_document(file_path: str | Path) -> list[Document]:
    """Load one file into LangChain `Document` objects.

    Raises `IngestionError` if the file is missing, its extension isn't
    supported, or the underlying loader fails or returns nothing.
    """

    path = Path(file_path)

    if not path.exists():
        raise IngestionError(f"File not found: {path}", details={"path": str(path)})

    build_loader = _LOADER_FACTORY_BY_EXTENSION.get(path.suffix.lower())
    if build_loader is None:
        raise IngestionError(
            f"Unsupported file type: {path.suffix!r}",
            details={"path": str(path), "supported_extensions": sorted(_LOADER_FACTORY_BY_EXTENSION)},
        )

    logger.info("Loading %s with %s", path.name, build_loader.__name__)

    try:
        documents = build_loader(str(path)).load()
    except Exception as exc:  # noqa: BLE001 - normalize every loader's own exception type
        raise IngestionError(f"Failed to load {path.name}: {exc}", details={"path": str(path)}) from exc

    if not documents:
        raise IngestionError(f"Loader produced no content for {path.name}", details={"path": str(path)})

    return documents


def supported_extensions() -> list[str]:
    """Extensions this module knows how to load, e.g. ['.docx', '.html', '.md', '.pdf']."""

    return sorted(_LOADER_FACTORY_BY_EXTENSION)
