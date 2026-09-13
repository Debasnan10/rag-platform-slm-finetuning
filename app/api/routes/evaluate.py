"""POST /evaluate - run the evaluation suite and return a metrics report.

Not yet implemented: depends on the evaluator, which doesn't exist yet.
Not wired into `app.api.main` until then.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/evaluate", tags=["evaluate"])

# TODO: POST / -> run app.evaluation.evaluator over the test set, return
# BLEU/ROUGE/Perplexity scores.
