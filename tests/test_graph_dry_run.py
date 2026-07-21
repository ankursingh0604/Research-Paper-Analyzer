"""
Dry-run test: mocks app.llm.structured_call so we can validate the graph's
control flow (fan-out, retry loops, fan-in barrier, termination) WITHOUT
spending real API calls. Not part of the deliverable's test suite - just a
sanity check used during development.
"""
import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.schemas import (
    CitationExtraction, Citation, ExecutiveSummary, KeyInsights,
    PaperAnalysis, PaperMetadata, ReviewResult,
)

call_counts = {"analysis": 0, "summary": 0, "citations": 0, "insights": 0, "review": 0}


def fake_structured_call(prompt, schema, model_name=None):
    if schema is PaperMetadata:
        return PaperMetadata(title="Attention Is All You Need (fake)", authors=["A. Vaswani"], year="2017", venue="NeurIPS")
    if schema is PaperAnalysis:
        call_counts["analysis"] += 1
        return PaperAnalysis(
            problem_statement="Test problem",
            methodology="Test method",
            key_experiments=["exp1"],
            main_findings=["finding1"],
        )
    if schema is ExecutiveSummary:
        call_counts["summary"] += 1
        return ExecutiveSummary(summary="A" * 50, word_count=50)
    if schema is CitationExtraction:
        call_counts["citations"] += 1
        return CitationExtraction(citations=[Citation(reference="Ref 1")], total_found=1)
    if schema is KeyInsights:
        call_counts["insights"] += 1
        return KeyInsights(takeaways=["t1"], implications=["i1"], potential_applications=["a1"])
    if schema is ReviewResult:
        call_counts["review"] += 1
        return ReviewResult(score=8, feedback="looks good", accuracy_ok=True, completeness_ok=True)
    raise AssertionError(f"Unexpected schema: {schema}")


# Force exactly one retry on the summary branch to prove retry+resume works,
# by scoring low the first time review_summary runs.
_summary_review_calls = {"n": 0}


def fake_structured_call_with_retry(prompt, schema, model_name=None):
    if schema is ReviewResult and 'following "summary" agent output' in prompt:
        _summary_review_calls["n"] += 1
        if _summary_review_calls["n"] == 1:
            return ReviewResult(score=4, feedback="Too short, add more detail.", accuracy_ok=True, completeness_ok=False)
        return ReviewResult(score=9, feedback="Better now.", accuracy_ok=True, completeness_ok=True)
    return fake_structured_call(prompt, schema, model_name)


def main():
    with patch("app.llm.structured_call", side_effect=fake_structured_call_with_retry):
        from app.graph import build_graph
        graph = build_graph()
        result = graph.invoke({
            "paper_text": "This is a fake paper about transformers and attention mechanisms.",
            "max_retries": 2,
            "quality_threshold": 7,
        })

    print("=== Final state keys ===", list(result.keys()))
    print("\n=== Review scores ===", result.get("review_scores"))
    print("\n=== Retry counts ===", result.get("retry_counts"))
    print("\n=== Call counts ===", call_counts)
    print("\n=== Log ===")
    for line in result.get("log", []):
        print(" -", line)
    print("\n=== Final brief (first 500 chars) ===")
    print(result.get("final_brief", "")[:500])

    assert result.get("final_brief"), "No final brief produced!"
    assert result["retry_counts"].get("summary", 0) == 1, "Expected exactly 1 retry on summary branch"
    assert result["review_scores"]["analysis"] >= 7
    assert result["review_scores"]["citations"] >= 7
    assert result["review_scores"]["insights"] >= 7
    print("\nALL ASSERTIONS PASSED")


if __name__ == "__main__":
    main()
