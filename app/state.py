"""
Shared state schema passed between every node in the LangGraph workflow.

Design notes:
- `retry_counts` / `review_scores` are keyed by agent name ("analysis",
  "summary", "citations", "insights") so the same review/routing logic
  can be reused generically for each branch.
- `errors` accumulates non-fatal problems (e.g. an agent hitting the
  retry ceiling) so the Boss Agent can surface them in the final brief
  instead of silently swallowing them.
"""
from __future__ import annotations

import operator
from typing import Annotated, Any, Optional, TypedDict


def merge_dicts(a: dict, b: dict) -> dict:
    """Reducer for dict-valued state keys that multiple parallel branches write to
    concurrently (e.g. the summary/insights branch and the citation branch both
    update `review_scores` in the same superstep). Shallow-merges; safe here since
    each branch only ever writes its own key (e.g. "summary" vs "citations")."""
    merged = dict(a)
    merged.update(b)
    return merged


class ResearchState(TypedDict, total=False):
    # --- input ---
    paper_text: str

    # --- agent outputs (dicts mirroring the Pydantic schemas in schemas.py) ---
    metadata: dict
    analysis: dict
    summary: dict
    citations: dict
    insights: dict

    # --- quality control ---
    # Concurrent branches (summary/insights vs citations) can update these in the
    # same superstep, so they need reducers instead of plain last-value overwrite.
    review_scores: Annotated[dict[str, int], merge_dicts]
    review_feedback: Annotated[dict[str, str], merge_dicts]
    retry_counts: Annotated[dict[str, int], merge_dicts]
    max_retries: int
    quality_threshold: int

    # --- orchestration / observability ---
    current_stage: str
    errors: Annotated[list[str], operator.add]
    log: Annotated[list[str], operator.add]   # human-readable trace of what happened, for the UI

    # --- final output ---
    final_brief: str
