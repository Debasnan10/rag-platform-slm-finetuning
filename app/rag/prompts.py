"""Prompt template for grounded question answering."""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate

RAG_PROMPT = PromptTemplate.from_template(
    "Answer the question using only the context below. If the context "
    "doesn't contain the answer, say you don't know - do not make one up.\n\n"
    "Context:\n{context}\n\n"
    "Question: {question}\n\n"
    "Answer:"
)


def format_context(documents: list[Document]) -> str:
    """Join retrieved chunks into a single context block, one per paragraph."""

    return "\n\n".join(document.page_content for document in documents)
