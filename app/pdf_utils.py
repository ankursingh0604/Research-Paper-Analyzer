"""
PDF / text ingestion for the research paper analyzer.

Supports:
- Local PDF file path
- Remote PDF URL (e.g. an arXiv PDF link)
- Raw pasted text (bypasses PDF parsing entirely)
"""
from __future__ import annotations

import io
import logging
import re

import pdfplumber
import requests

logger = logging.getLogger(__name__)

MAX_CHARS = 60_000  # keep well within context window across all agent calls


def extract_text_from_pdf_bytes(pdf_bytes: bytes) -> str:
    """Extract and lightly clean text from raw PDF bytes, preserving page order."""
    text_parts: list[str] = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text() or ""
            if page_text.strip():
                text_parts.append(f"\n--- Page {i + 1} ---\n{page_text}")
    raw = "\n".join(text_parts)
    return _clean_text(raw)


def extract_text_from_pdf_path(path: str) -> str:
    with open(path, "rb") as f:
        return extract_text_from_pdf_bytes(f.read())


def extract_text_from_url(url: str) -> str:
    """Download a PDF (e.g. an arXiv link) and extract its text."""
    logger.info("Downloading PDF from %s", url)
    resp = requests.get(url, timeout=30, headers={"User-Agent": "research-paper-analyzer/1.0"})
    resp.raise_for_status()
    content_type = resp.headers.get("Content-Type", "")
    if "pdf" not in content_type and not url.lower().endswith(".pdf"):
        logger.warning("URL does not look like a PDF (Content-Type=%s); attempting extraction anyway", content_type)
    return extract_text_from_pdf_bytes(resp.content)


def _clean_text(text: str) -> str:
    # Collapse excessive whitespace/newlines that hurt LLM context efficiency
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    if len(text) > MAX_CHARS:
        logger.warning("Paper text truncated from %d to %d chars to fit context window", len(text), MAX_CHARS)
        text = text[:MAX_CHARS] + "\n\n[...truncated...]"
    return text


def load_paper(*, pdf_path: str | None = None, pdf_url: str | None = None, raw_text: str | None = None) -> str:
    """Single entry point: exactly one of pdf_path / pdf_url / raw_text should be given."""
    if raw_text and raw_text.strip():
        return _clean_text(raw_text)
    if pdf_path:
        return extract_text_from_pdf_path(pdf_path)
    if pdf_url:
        return extract_text_from_url(pdf_url)
    raise ValueError("Provide one of pdf_path, pdf_url, or raw_text")
