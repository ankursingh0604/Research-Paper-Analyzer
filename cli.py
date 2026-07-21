"""
CLI entry point.

Usage:
    python cli.py --pdf path/to/paper.pdf
    python cli.py --url https://arxiv.org/pdf/1706.03762
    python cli.py --text "paste raw paper text here"

Writes the generated research brief to outputs/research_brief.md and also
prints it to stdout.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    parser = argparse.ArgumentParser(description="AI-Powered Research Paper Analyzer")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--pdf", help="Path to a local PDF file")
    src.add_argument("--url", help="URL to a PDF (e.g. an arXiv paper)")
    src.add_argument("--text", help="Raw paper text (bypasses PDF parsing)")
    parser.add_argument("--out", default="outputs/research_brief.md", help="Output markdown file path")
    args = parser.parse_args()

    from app.pdf_utils import load_paper
    from app.graph import build_graph

    print("Loading paper...")
    paper_text = load_paper(pdf_path=args.pdf, pdf_url=args.url, raw_text=args.text)
    print(f"Loaded {len(paper_text)} characters of paper text.\n")

    print("Running multi-agent analysis pipeline (this calls the Gemini API several times)...\n")
    graph = build_graph()
    result = graph.invoke({
        "paper_text": paper_text,
        "max_retries": int(os.environ.get("MAX_RETRIES", 2)),
        "quality_threshold": int(os.environ.get("QUALITY_THRESHOLD", 7)),
    })

    print("--- Workflow trace ---")
    for line in result.get("log", []):
        print(" ", line)
    print()

    if result.get("errors"):
        print("--- Warnings / non-fatal errors ---")
        for err in result["errors"]:
            print(" ", err)
        print()

    brief = result.get("final_brief", "")
    if not brief:
        print("ERROR: no research brief was produced. Check the trace above.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(brief)

    print(f"Research brief written to {args.out}\n")
    print("=" * 70)
    print(brief)


if __name__ == "__main__":
    main()
