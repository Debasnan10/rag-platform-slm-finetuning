"""POST /query - ask a question against the RAG pipeline.

Placeholder for now - see app/api/routes/ingest.py for why. Implemented
in Phase 9 on top of the RAG chain built in Phase 4.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/query", tags=["query"])

# TODO(Phase 9): POST / -> run app.rag.chain against the question, return
# the answer plus source citations.
