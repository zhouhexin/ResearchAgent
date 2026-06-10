"""Request and response schemas for the QA API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """User-facing ask request."""

    query: str = Field(..., min_length=1, max_length=2000)


class PaperLink(BaseModel):
    """Local PDF link matched from a public QA answer."""

    id: str
    title: str
    preview_url: str
    download_url: str


class AskResponse(BaseModel):
    """User-facing ask response.

    Only the final answer is returned to the frontend. `run_id` is included for
    backend troubleshooting but does not expose retrieval or context details.
    """

    answer: str
    run_id: Optional[str] = None
    paper_links: list[PaperLink] = Field(default_factory=list)
    error: Optional[str] = None


class FeedbackRequest(BaseModel):
    """User feedback for one public QA answer."""

    run_id: Optional[str] = None
    query: str = Field(..., min_length=1, max_length=2000)
    answer: str = Field(..., min_length=1)
    rating: str = Field(..., pattern="^(accurate|inaccurate)$")


class FeedbackResponse(BaseModel):
    """Feedback write result."""

    ok: bool
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
