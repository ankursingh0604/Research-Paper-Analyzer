"""
Wires together the LangGraph state machine described in the assignment's
architecture diagram:

    Input
      |
    Boss Agent (init) --------------------+
      |                                   |
    Paper Analyzer -> Review -> [retry?]  Metadata Agent
      | (approved)                          |
      +--> Summary Generator -> Review -> [retry?] -> Key Insights -> Review -> [retry?] --+
      |                                                                                     |
      +--> Citation Extractor -> Review -> [retry?] -----------------------------------+   |
                                                                                        v   v
                                                                              Boss Agent (combine)
                                                                                        |
                                                                                     Output

Design choices:
  - Metadata extraction runs in parallel with the Paper Analyzer since it's
    independent (doesn't need the analysis to run).
  - Citation Extractor runs in parallel with Summary Generator (both only
    need the approved analysis + paper text).
  - Key Insights runs *after* the Summary is approved, because its prompt
    is grounded in the summary text - this is a deliberate sequential
    dependency, not an oversight.
  - The Boss "combine" node has multiple incoming edges (metadata,
    insights-branch, citations-branch) - LangGraph's Pregel engine treats
    this as a fan-in barrier, so `boss_combine` only runs once every
    upstream branch has produced (or exhausted retries on) its output.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, START, END

from . import agents
from .state import ResearchState


def build_graph():
    g = StateGraph(ResearchState)

    # --- nodes ---
    g.add_node("boss_init", agents.boss_init)
    g.add_node("metadata_agent", agents.metadata_agent)

    g.add_node("paper_analyzer", agents.paper_analyzer_agent)
    g.add_node("review_analysis", agents.make_review_node("analysis"))
    g.add_node("retry_analysis", agents.bump_retry_counter("analysis"))

    g.add_node("summary_agent", agents.summary_agent)
    g.add_node("review_summary", agents.make_review_node("summary"))
    g.add_node("retry_summary", agents.bump_retry_counter("summary"))

    g.add_node("citation_agent", agents.citation_agent)
    g.add_node("review_citations", agents.make_review_node("citations"))
    g.add_node("retry_citations", agents.bump_retry_counter("citations"))

    g.add_node("insights_agent", agents.insights_agent)
    g.add_node("review_insights", agents.make_review_node("insights"))
    g.add_node("retry_insights", agents.bump_retry_counter("insights"))

    g.add_node("boss_combine", agents.boss_combine)

    # --- entry: fan out to metadata (independent) and the analyzer ---
    g.add_edge(START, "boss_init")
    g.add_edge("boss_init", "metadata_agent")
    g.add_edge("boss_init", "paper_analyzer")

    # metadata has no review step - feeds straight into the final barrier
    g.add_edge("metadata_agent", "boss_combine")

    # --- analysis branch (gate for everything downstream) ---
    # Approval fans out to BOTH the Summary and Citation branches at once.
    g.add_edge("paper_analyzer", "review_analysis")
    g.add_conditional_edges(
        "review_analysis",
        agents.make_route_after_review(
            "analysis",
            retry_node="retry_analysis",
            next_node=["summary_agent", "citation_agent"],
        ),
    )
    g.add_edge("retry_analysis", "paper_analyzer")

    # --- summary branch -> feeds key insights (sequential dependency) ---
    g.add_edge("summary_agent", "review_summary")
    g.add_conditional_edges(
        "review_summary",
        agents.make_route_after_review("summary", retry_node="retry_summary", next_node="insights_agent"),
    )
    g.add_edge("retry_summary", "summary_agent")

    g.add_edge("insights_agent", "review_insights")
    g.add_conditional_edges(
        "review_insights",
        agents.make_route_after_review("insights", retry_node="retry_insights", next_node="boss_combine"),
    )
    g.add_edge("retry_insights", "insights_agent")

    # --- citation branch (runs in parallel with summary+insights) ---
    g.add_edge("citation_agent", "review_citations")
    g.add_conditional_edges(
        "review_citations",
        agents.make_route_after_review("citations", retry_node="retry_citations", next_node="boss_combine"),
    )
    g.add_edge("retry_citations", "citation_agent")

    g.add_edge("boss_combine", END)

    return g.compile()
