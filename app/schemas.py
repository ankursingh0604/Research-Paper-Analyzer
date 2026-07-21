"""
Structured-output schemas for every agent in the pipeline.

Gemini's JSON mode is given a cleaned JSON schema (via `to_gemini_schema()`
in llm.py) derived from these Pydantic models, so the model is forced to
return valid JSON that maps directly onto them. This removes the need for
brittle regex/string parsing of LLM output.

NOTE: these models keep `default` / `default_factory` values for normal
Pydantic parsing convenience. Gemini's response_schema does NOT support
the `default` keyword, so llm.py strips it out of the generated JSON
schema before sending it to the API - it does not need to be removed here.
"""
from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


class PaperMetadata(BaseModel):
    title: str = Field(description="Full title of the paper")
    authors: List[str] = Field(default_factory=list, description="Author names")
    year: Optional[str] = Field(default=None, description="Publication year, if identifiable")
    venue: Optional[str] = Field(default=None, description="Conference/journal/venue, if identifiable")


class PaperAnalysis(BaseModel):
    problem_statement: str = Field(description="The core problem/research question the paper addresses")
    methodology: str = Field(description="The approach, model, or method used")
    key_experiments: List[str] = Field(default_factory=list, description="Main experiments/evaluations run")
    main_findings: List[str] = Field(default_factory=list, description="Key results and findings")
    hypothesis: Optional[str] = Field(default=None, description="Central hypothesis being tested, if any")


class ExecutiveSummary(BaseModel):
    summary: str = Field(description="A 150-200 word executive summary covering problem, approach, and results")
    word_count: int = Field(description="Approximate word count of the summary")


class Citation(BaseModel):
    reference: str = Field(description="Citation text as it appears / can be reconstructed (authors, title, venue, year)")
    relevance: Optional[str] = Field(default=None, description="Why this reference matters to the paper's argument")


class CitationExtraction(BaseModel):
    citations: List[Citation] = Field(default_factory=list)
    total_found: int = Field(description="Total number of distinct citations identified")


class KeyInsights(BaseModel):
    takeaways: List[str] = Field(default_factory=list, description="Actionable, practical takeaways")
    implications: List[str] = Field(default_factory=list, description="Broader implications of the work")
    potential_applications: List[str] = Field(default_factory=list, description="Real-world application ideas")


class ReviewResult(BaseModel):
    score: int = Field(description="Quality score from 1 (poor) to 10 (excellent)")
    feedback: str = Field(description="Specific, actionable feedback explaining the score")
    accuracy_ok: bool = Field(description="Whether the content appears faithful to the source paper (no hallucination)")
    completeness_ok: bool = Field(description="Whether the content covers what was asked for completely")