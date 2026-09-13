"""POST /query - ask a question against the RAG pipeline.

Not yet implemented: depends on the RAG chain, which doesn't exist yet.
Not wired into `app.api.main` until then.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/query", tags=["query"])

# TODO: POST / -> run app.rag.chain against the question, return the
# answer plus source citations.
