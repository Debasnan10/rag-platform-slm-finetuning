"""POST /ingest - document ingestion route.

Placeholder for now: the folder structure is laid down in Phase 1 so the
whole project shape is visible from day one, but the actual endpoint is
implemented in Phase 9, once ingestion (Phase 2) and the FAISS store
(Phase 3) exist for it to call into. Not yet wired into `app.api.main`.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/ingest", tags=["ingest"])

# TODO(Phase 9): POST / -> run app.ingestion.pipeline against an uploaded
# or referenced document, then add the resulting chunks to the FAISS store.
