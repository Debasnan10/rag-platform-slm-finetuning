"""POST /ingest - document ingestion route.

Not yet implemented: the vector store this route needs to write into
doesn't exist yet. Not wired into `app.api.main` until then.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/ingest", tags=["ingest"])

# TODO: POST / -> run app.ingestion.pipeline against an uploaded or
# referenced document, then add the resulting chunks to the FAISS store.
