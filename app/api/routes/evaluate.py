"""POST /evaluate - run the evaluation suite and return a metrics report.

Placeholder for now - see app/api/routes/ingest.py for why. Implemented
in Phase 9 on top of the evaluator built in Phase 8.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/evaluate", tags=["evaluate"])

# TODO(Phase 9): POST / -> run app.evaluation.evaluator over the test set,
# return BLEU/ROUGE/Perplexity scores.
