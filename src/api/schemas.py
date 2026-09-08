"""Request/response models for the public API.

The response is deliberately lean -- just the answer, citations, and refusal
info. Raw retrieval scores and chunk text are dropped entirely; they're
implementation detail the eval harness needs, not something the public-facing
demo should expose.
"""

from enum import Enum

from pydantic import BaseModel, Field


class RetrievalMode(str, Enum):
    baseline = "baseline"
    hybrid = "hybrid"


class QueryRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    retrieval_mode: RetrievalMode = RetrievalMode.hybrid


class Citation(BaseModel):
    issue_number: int
    url: str


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    refused: bool
    refusal_reason: str | None
