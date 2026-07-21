"""
Node functions for the LangGraph workflow. Each function takes the shared
ResearchState, does its work, and returns a partial state update (LangGraph
merges dict returns into state).

Every generating agent follows the same pattern:
    1. Build a prompt (including feedback from a prior failed review, if any)
    2. Call the LLM for a structured Pydantic object
    3. Store it in state, log the step
    4. On LLMCallError: record the error, leave prior output (if any) in place
       so a retry has something to build on, and let the review step fail it
       naturally rather than crashing the whole graph run.

The Review Agent is generic - it takes an `agent_name` + the JSON blob to
review, so the same function backs every review edge in the graph.
"""
from __future__ import annotations

import json
import logging
from functools import partial

from . import prompts
from .llm import LLMCallError, structured_call
from .schemas import (
    CitationExtraction,
    ExecutiveSummary,
    KeyInsights,
    PaperAnalysis,
    PaperMetadata,
    ReviewResult,
)
from .state import ResearchState

logger = logging.getLogger(__name__)


def _log(state: ResearchState, message: str) -> list[str]:
    # `log` uses an operator.add reducer in ResearchState, so we return only the
    # new delta entry - LangGraph appends it to the accumulated list automatically.
    logger.info(message)
    return [message]


def _errors(state: ResearchState, message: str) -> list[str]:
    # Same delta pattern as `_log` - `errors` also uses an operator.add reducer.
    return [message]


# ---------------------------------------------------------------------------
# Boss Agent - orchestrator
# ---------------------------------------------------------------------------

def boss_init(state: ResearchState) -> dict:
    """Entry point: sets defaults and kicks off the workflow."""
    return {
        "review_scores": {},
        "review_feedback": {},
        "retry_counts": {},
        "max_retries": state.get("max_retries", 2),
        "quality_threshold": state.get("quality_threshold", 7),
        "errors": [],
        "current_stage": "metadata",
        "log": _log(state, "Boss Agent: workflow started, delegating to Metadata + Paper Analyzer."),
    }


_REQUIRED_FOR_COMBINE = ("analysis", "summary", "citations", "insights")


def boss_combine(state: ResearchState) -> dict:
    """
    Final step: assemble everything into one research brief.

    IMPORTANT: `boss_combine` has three incoming edges (metadata, the
    insights-branch, and the citations-branch) that complete at different
    times - metadata finishes almost immediately, while the other two may
    loop through retries first. LangGraph triggers a node whenever *any*
    predecessor arrives (it is not an implicit AND-barrier across
    supersteps), so this node can be invoked multiple times before every
    branch is actually done. We guard against assembling (and overwriting
    `final_brief`) with incomplete data by checking that every required
    piece of state is present; if not, this call is a no-op and we simply
    wait for the next arrival.
    """
    scores = state.get("review_scores", {})
    if state.get("metadata") is None or any(agent not in scores for agent in _REQUIRED_FOR_COMBINE):
        return {"log": _log(state, "Boss Agent: combine triggered but branches still in flight, waiting...")}

    meta = state.get("metadata", {})
    analysis = state.get("analysis", {})
    summary = state.get("summary", {})
    citations = state.get("citations", {})
    insights = state.get("insights", {})
    scores = state.get("review_scores", {})
    errors = state.get("errors", [])

    lines = []
    lines.append(f"# Research Brief: {meta.get('title', 'Untitled Paper')}\n")

    authors = meta.get("authors") or []
    if authors:
        lines.append(f"**Authors:** {', '.join(authors)}")
    if meta.get("year"):
        lines.append(f"**Year:** {meta['year']}")
    if meta.get("venue"):
        lines.append(f"**Venue:** {meta['venue']}")
    lines.append("")

    lines.append("## Research Analysis")
    lines.append(f"**Problem Statement:** {analysis.get('problem_statement', 'N/A')}\n")
    lines.append(f"**Methodology:** {analysis.get('methodology', 'N/A')}\n")
    if analysis.get("hypothesis"):
        lines.append(f"**Hypothesis:** {analysis['hypothesis']}\n")
    if analysis.get("key_experiments"):
        lines.append("**Key Experiments:**")
        lines.extend(f"- {e}" for e in analysis["key_experiments"])
        lines.append("")
    if analysis.get("main_findings"):
        lines.append("**Main Findings:**")
        lines.extend(f"- {f}" for f in analysis["main_findings"])
        lines.append("")

    lines.append("## Executive Summary")
    lines.append(summary.get("summary", "N/A"))
    lines.append("")

    lines.append("## Citations & References")
    cite_list = citations.get("citations", [])
    if cite_list:
        for c in cite_list:
            ref = c.get("reference", "")
            rel = c.get("relevance")
            lines.append(f"- {ref}" + (f" _(relevance: {rel})_" if rel else ""))
    else:
        lines.append("_No citations could be extracted from the available text._")
    lines.append("")

    lines.append("## Key Insights")
    if insights.get("takeaways"):
        lines.append("**Takeaways:**")
        lines.extend(f"- {t}" for t in insights["takeaways"])
    if insights.get("implications"):
        lines.append("\n**Implications:**")
        lines.extend(f"- {i}" for i in insights["implications"])
    if insights.get("potential_applications"):
        lines.append("\n**Potential Applications:**")
        lines.extend(f"- {a}" for a in insights["potential_applications"])
    lines.append("")

    lines.append("## Quality Review Scores")
    for agent_name, score in scores.items():
        lines.append(f"- {agent_name}: {score}/10")

    if errors:
        lines.append("\n## Known Issues")
        lines.extend(f"- {e}" for e in errors)

    brief = "\n".join(lines)
    return {
        "final_brief": brief,
        "current_stage": "done",
        "log": _log(state, "Boss Agent: all branches approved (or retry-exhausted), final brief assembled."),
    }


# ---------------------------------------------------------------------------
# Sub-agents
# ---------------------------------------------------------------------------

def metadata_agent(state: ResearchState) -> dict:
    prompt = prompts.METADATA_PROMPT.format(paper_text=state["paper_text"])
    try:
        result = structured_call(prompt, PaperMetadata)
        return {"metadata": result.model_dump(), "log": _log(state, "Metadata agent: extracted paper metadata.")}
    except LLMCallError as exc:
        return {
            "metadata": {"title": "Unknown Title", "authors": [], "year": None, "venue": None},
            "errors": _errors(state, f"Metadata extraction failed: {exc}"),
            "log": _log(state, f"Metadata agent: FAILED ({exc})"),
        }


def paper_analyzer_agent(state: ResearchState) -> dict:
    fb = prompts.feedback_block(state.get("review_feedback", {}).get("analysis"))
    prompt = prompts.ANALYZER_PROMPT.format(paper_text=state["paper_text"], feedback_block=fb)
    try:
        result = structured_call(prompt, PaperAnalysis)
        return {"analysis": result.model_dump(), "log": _log(state, "Paper Analyzer agent: produced analysis draft.")}
    except LLMCallError as exc:
        return {
            "errors": _errors(state, f"Paper Analyzer failed: {exc}"),
            "log": _log(state, f"Paper Analyzer agent: FAILED ({exc})"),
        }


def summary_agent(state: ResearchState) -> dict:
    fb = prompts.feedback_block(state.get("review_feedback", {}).get("summary"))
    prompt = prompts.SUMMARY_PROMPT.format(
        paper_text=state["paper_text"],
        analysis_json=json.dumps(state.get("analysis", {}), indent=2),
        feedback_block=fb,
    )
    try:
        result = structured_call(prompt, ExecutiveSummary)
        return {"summary": result.model_dump(), "log": _log(state, "Summary Generator agent: produced summary draft.")}
    except LLMCallError as exc:
        return {
            "errors": _errors(state, f"Summary Generator failed: {exc}"),
            "log": _log(state, f"Summary Generator agent: FAILED ({exc})"),
        }


def citation_agent(state: ResearchState) -> dict:
    fb = prompts.feedback_block(state.get("review_feedback", {}).get("citations"))
    prompt = prompts.CITATION_PROMPT.format(paper_text=state["paper_text"], feedback_block=fb)
    try:
        result = structured_call(prompt, CitationExtraction)
        return {"citations": result.model_dump(), "log": _log(state, "Citation Extractor agent: produced citations draft.")}
    except LLMCallError as exc:
        return {
            "errors": _errors(state, f"Citation Extractor failed: {exc}"),
            "log": _log(state, f"Citation Extractor agent: FAILED ({exc})"),
        }


def insights_agent(state: ResearchState) -> dict:
    fb = prompts.feedback_block(state.get("review_feedback", {}).get("insights"))
    prompt = prompts.INSIGHTS_PROMPT.format(
        analysis_json=json.dumps(state.get("analysis", {}), indent=2),
        summary_text=state.get("summary", {}).get("summary", ""),
        feedback_block=fb,
    )
    try:
        result = structured_call(prompt, KeyInsights)
        return {"insights": result.model_dump(), "log": _log(state, "Key Insights agent: produced insights draft.")}
    except LLMCallError as exc:
        return {
            "errors": _errors(state, f"Key Insights failed: {exc}"),
            "log": _log(state, f"Key Insights agent: FAILED ({exc})"),
        }


# ---------------------------------------------------------------------------
# Review Agent (generic - reused for every branch)
# ---------------------------------------------------------------------------

_STATE_KEY_FOR_AGENT = {
    "analysis": "analysis",
    "summary": "summary",
    "citations": "citations",
    "insights": "insights",
}


def make_review_node(agent_name: str):
    """Factory: returns a review node function bound to a specific agent's output."""

    def _review(state: ResearchState) -> dict:
        # review_scores/review_feedback use a merge_dicts reducer, so we return
        # only this agent's key (a "delta"), not a full-dict copy - this avoids
        # concurrent branches (e.g. summary vs citations) racing on a stale
        # snapshot of the whole dict.
        state_key = _STATE_KEY_FOR_AGENT[agent_name]
        output = state.get(state_key)

        if not output:
            # Upstream generation failed entirely - fail the review without calling the LLM.
            return {
                "review_scores": {agent_name: 0},
                "review_feedback": {agent_name: "No output was produced to review (generation failed)."},
                "log": _log(state, f"Review Agent: {agent_name} has no output to review, scoring 0."),
            }

        prompt = prompts.REVIEW_PROMPT.format(
            agent_name=agent_name,
            paper_text=state["paper_text"],
            output_json=json.dumps(output, indent=2),
        )
        try:
            result = structured_call(prompt, ReviewResult)
            return {
                "review_scores": {agent_name: result.score},
                "review_feedback": {agent_name: result.feedback},
                "log": _log(
                    state,
                    f"Review Agent: {agent_name} scored {result.score}/10 "
                    f"(accuracy_ok={result.accuracy_ok}, completeness_ok={result.completeness_ok}).",
                ),
            }
        except LLMCallError as exc:
            # If the reviewer itself fails, don't block the pipeline forever -
            # accept the draft as-is but flag it for human attention.
            return {
                "review_scores": {agent_name: state.get("quality_threshold", 7)},
                "review_feedback": {agent_name: "Review call failed; auto-accepted without review."},
                "errors": _errors(state, f"Review Agent failed for {agent_name}: {exc}"),
                "log": _log(state, f"Review Agent: FAILED reviewing {agent_name} ({exc}); auto-accepted."),
            }

    _review.__name__ = f"review_{agent_name}"
    return _review


def make_route_after_review(agent_name: str, retry_node: str, next_node):
    """
    Factory: returns a conditional-edge routing function for the given
    branch. Retries the generating agent up to max_retries if the score is
    below threshold; otherwise (or once retries are exhausted) proceeds.

    `next_node` may be a single node name (str) or a list of node names,
    for fan-out to multiple parallel branches at once. Returned values are
    real node names, consumed directly by add_conditional_edges.
    """

    def _route(state: ResearchState):
        threshold = state.get("quality_threshold", 7)
        score = state.get("review_scores", {}).get(agent_name, 0)
        retries = dict(state.get("retry_counts", {}))
        attempts_so_far = retries.get(agent_name, 0)
        max_retries = state.get("max_retries", 2)

        if score >= threshold:
            return next_node

        if attempts_so_far < max_retries:
            return retry_node

        return next_node  # retries exhausted - force-accept and move on

    return _route


def bump_retry_counter(agent_name: str):
    """Node wrapper: increments the retry counter for an agent right before it re-runs."""

    def _bump(state: ResearchState) -> dict:
        new_count = state.get("retry_counts", {}).get(agent_name, 0) + 1
        return {
            # Delta-only update - see note in make_review_node about merge_dicts.
            "retry_counts": {agent_name: new_count},
            "log": _log(state, f"Boss Agent: {agent_name} scored below threshold, retrying (attempt {new_count})."),
        }

    return _bump
