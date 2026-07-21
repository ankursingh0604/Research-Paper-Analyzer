"""
Streamlit UI for the AI-Powered Research Paper Analyzer.

Run with: streamlit run streamlit_app.py

Uses graph.stream() (instead of .invoke()) so we can show live progress as
each agent node fires - which agent is currently running, review scores as
they come in, and the retry/iteration history - rather than a single
blocking spinner.
"""
from __future__ import annotations

import logging
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO)

from app.graph import build_graph
from app.pdf_utils import load_paper

st.set_page_config(page_title="Research Paper Analyzer", page_icon="\U0001F4C4", layout="wide")

AGENT_DISPLAY_NAMES = {
    "boss_init": "Boss Agent (starting workflow)",
    "metadata_agent": "Metadata Agent",
    "paper_analyzer": "Paper Analyzer Agent",
    "review_analysis": "Review Agent \u2192 Analysis",
    "retry_analysis": "Retrying Paper Analyzer...",
    "summary_agent": "Summary Generator Agent",
    "review_summary": "Review Agent \u2192 Summary",
    "retry_summary": "Retrying Summary Generator...",
    "citation_agent": "Citation Extractor Agent",
    "review_citations": "Review Agent \u2192 Citations",
    "retry_citations": "Retrying Citation Extractor...",
    "insights_agent": "Key Insights Agent",
    "review_insights": "Review Agent \u2192 Key Insights",
    "retry_insights": "Retrying Key Insights...",
    "boss_combine": "Boss Agent (combining final brief)",
}

PIPELINE_STAGES = [
    "boss_init", "metadata_agent", "paper_analyzer", "review_analysis",
    "summary_agent", "review_summary", "citation_agent", "review_citations",
    "insights_agent", "review_insights", "boss_combine",
]


def run_pipeline(paper_text: str, max_retries: int, threshold: int):
    graph = build_graph()

    status_area = st.status("Starting multi-agent workflow...", expanded=True)
    progress_bar = st.progress(0)
    score_cols = st.columns(4)
    score_placeholders = {name: score_cols[i].empty() for i, name in enumerate(["analysis", "summary", "citations", "insights"])}

    final_state: dict = {}
    seen_stage_idx = 0

    for update in graph.stream(
        {"paper_text": paper_text, "max_retries": max_retries, "quality_threshold": threshold},
        stream_mode="updates",
    ):
        for node_name, partial in update.items():
            label = AGENT_DISPLAY_NAMES.get(node_name, node_name)
            status_area.update(label=label)
            if partial.get("log"):
                for line in partial["log"]:
                    status_area.write(f"- {line}")

            final_state.update({k: v for k, v in partial.items() if k not in ("log",)})
            # merge dict-valued deltas properly for display purposes
            if "review_scores" in partial:
                final_state.setdefault("review_scores", {})
                final_state["review_scores"] = {**final_state.get("review_scores", {}), **partial["review_scores"]}
                for agent, score in final_state["review_scores"].items():
                    if agent in score_placeholders:
                        color = "\U0001F7E2" if score >= threshold else "\U0001F7E1"
                        score_placeholders[agent].metric(f"{color} {agent}", f"{score}/10")

            if node_name in PIPELINE_STAGES:
                idx = PIPELINE_STAGES.index(node_name)
                seen_stage_idx = max(seen_stage_idx, idx)
                progress_bar.progress(min(1.0, (seen_stage_idx + 1) / len(PIPELINE_STAGES)))

    status_area.update(label="Done!", state="complete", expanded=False)
    progress_bar.progress(1.0)
    return final_state


def main():
    st.title("\U0001F4C4 AI-Powered Research Paper Analyzer")
    st.caption(
        "Multi-agent system (LangGraph + OpenAI): Paper Analyzer \u2192 Summary / Citations / Key Insights, "
        "each gated by a Review Agent with automatic retry on low quality scores."
    )

    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("OPENAI_API_KEY is not set. Add it to a .env file (see .env.example) before running an analysis.")

    with st.sidebar:
        st.header("Settings")
        threshold = st.slider("Quality threshold (review score to pass)", 1, 10, int(os.environ.get("QUALITY_THRESHOLD", 7)))
        max_retries = st.slider("Max retries per agent", 0, 3, int(os.environ.get("MAX_RETRIES", 2)))

    tab_pdf, tab_url, tab_text = st.tabs(["Upload PDF", "Paper URL", "Paste text"])
    paper_text = None

    with tab_pdf:
        uploaded = st.file_uploader("Upload a research paper PDF", type=["pdf"])
        if uploaded and st.button("Analyze uploaded PDF", type="primary"):
            with st.spinner("Extracting text from PDF..."):
                from app.pdf_utils import extract_text_from_pdf_bytes
                paper_text = extract_text_from_pdf_bytes(uploaded.read())

    with tab_url:
        url = st.text_input("PDF URL (e.g. an arXiv link like https://arxiv.org/pdf/1706.03762)")
        if url and st.button("Analyze from URL", type="primary"):
            with st.spinner("Downloading and extracting PDF..."):
                paper_text = load_paper(pdf_url=url)

    with tab_text:
        raw = st.text_area("Paste paper text", height=200)
        if raw and st.button("Analyze pasted text", type="primary"):
            paper_text = load_paper(raw_text=raw)

    if paper_text:
        st.divider()
        st.subheader("Workflow Progress")
        result = run_pipeline(paper_text, max_retries, threshold)

        if result.get("errors"):
            with st.expander("\u26A0\uFE0F Warnings / non-fatal issues", expanded=False):
                for e in result["errors"]:
                    st.write("-", e)

        if result.get("final_brief"):
            st.divider()
            st.subheader("Research Brief")
            st.markdown(result["final_brief"])
            st.download_button(
                "Download brief as Markdown",
                data=result["final_brief"],
                file_name="research_brief.md",
                mime="text/markdown",
            )
        else:
            st.error("The pipeline did not produce a final brief. Check the workflow log above for details.")


if __name__ == "__main__":
    main()