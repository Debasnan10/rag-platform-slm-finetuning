"""End-to-end RAG chain: retrieve -> rerank -> prompt -> generate.

Ties together the pieces built in earlier phases - a `VectorStore` for
dense retrieval, `build_hybrid_retriever()` to blend in BM25, `rerank()` to
sharpen the candidate set with a cross-encoder, and an `LLMProvider` to
generate the final answer - into the single call a caller actually wants:
question in, grounded answer (plus the chunks it came from) out.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from langchain_core.documents import Document

from app.core.config import Settings, get_settings
from app.core.exceptions import LLMError
from app.llm.base import LLMProvider
from app.llm.ollama import get_ollama_provider
from app.rag.prompts import RAG_PROMPT, format_context
from app.retrieval.hybrid import build_hybrid_retriever
from app.retrieval.reranker import ScoreFn, rerank
from app.vectorstore.store import VectorStore, get_vector_store

_NO_CONTEXT_ANSWER = "I don't have enough information in the indexed documents to answer that."


@dataclass
class RAGResult:
    """An answer together with the chunks it was grounded in."""

    answer: str
    sources: list[Document] = field(default_factory=list)


class RAGChain:
    """Answers questions by retrieving, reranking, then generating from context only."""

    def __init__(
        self,
        documents: list[Document],
        vector_store: VectorStore | None = None,
        llm_provider: LLMProvider | None = None,
        settings: Settings | None = None,
        score_fn: ScoreFn | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._llm_provider = llm_provider or get_ollama_provider()
        self._score_fn = score_fn

        vector_store = vector_store or get_vector_store()
        self._retriever = build_hybrid_retriever(
            documents, vector_store=vector_store, settings=self._settings
        )

    def answer(self, question: str) -> RAGResult:
        """Retrieve context for `question`, then ask the LLM to answer from it alone."""

        candidates = self._retriever.invoke(question)
        top_chunks = rerank(
            question,
            candidates,
            top_n=self._settings.reranker_top_n,
            score_fn=self._score_fn,
        )

        if not top_chunks:
            return RAGResult(answer=_NO_CONTEXT_ANSWER, sources=[])

        prompt = RAG_PROMPT.format(context=format_context(top_chunks), question=question)

        try:
            llm = self._llm_provider.get_llm()
            answer_text = llm.invoke(prompt)
        except Exception as exc:  # noqa: BLE001 - normalize every backend's own exception type
            raise LLMError(f"Failed to generate an answer: {exc}") from exc

        return RAGResult(answer=str(answer_text).strip(), sources=top_chunks)
