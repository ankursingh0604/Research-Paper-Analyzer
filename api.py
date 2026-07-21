"""
Optional FastAPI wrapper around the LangGraph pipeline.

Run with: uvicorn api:app --reload

POST /analyze with a PDF file (multipart) or JSON {"url": ...} / {"text": ...}
returns the assembled research brief plus the full workflow trace.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from app.graph import build_graph
from app.pdf_utils import extract_text_from_pdf_bytes, load_paper

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Research Paper Analyzer",
    description="Multi-agent (LangGraph + Gemini) pipeline that turns a research paper into a structured brief.",
    version="1.0.0",
)


class TextOrUrlRequest(BaseModel):
    url: Optional[str] = None
    text: Optional[str] = None
    max_retries: int = int(os.environ.get("MAX_RETRIES", 2))
    quality_threshold: int = int(os.environ.get("QUALITY_THRESHOLD", 7))


class AnalyzeResponse(BaseModel):
    final_brief: str
    review_scores: dict
    retry_counts: dict
    errors: list[str]
    log: list[str]


def _run(paper_text: str, max_retries: int, quality_threshold: int) -> AnalyzeResponse:
    graph = build_graph()
    result = graph.invoke({
        "paper_text": paper_text,
        "max_retries": max_retries,
        "quality_threshold": quality_threshold,
    })
    if not result.get("final_brief"):
        raise HTTPException(status_code=500, detail={"message": "Pipeline failed to produce a brief", "log": result.get("log", []), "errors": result.get("errors", [])})
    return AnalyzeResponse(
        final_brief=result["final_brief"],
        review_scores=result.get("review_scores", {}),
        retry_counts=result.get("retry_counts", {}),
        errors=result.get("errors", []),
        log=result.get("log", []),
    )


@app.get("/health")
def health():
    return {"status": "ok", "gemini_key_configured": bool(os.environ.get("GOOGLE_API_KEY"))}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: TextOrUrlRequest):
    if not req.url and not req.text:
        raise HTTPException(status_code=400, detail="Provide either 'url' or 'text'.")
    try:
        paper_text = load_paper(pdf_url=req.url, raw_text=req.text)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to load paper: {exc}") from exc
    return _run(paper_text, req.max_retries, req.quality_threshold)


@app.post("/analyze/upload", response_model=AnalyzeResponse)
async def analyze_upload(file: UploadFile = File(...), max_retries: int = 2, quality_threshold: int = 7):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF uploads are supported.")
    pdf_bytes = await file.read()
    try:
        paper_text = extract_text_from_pdf_bytes(pdf_bytes)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to parse PDF: {exc}") from exc
    return _run(paper_text, max_retries, quality_threshold)
