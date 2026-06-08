"""Request and response schemas for the QA API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    """User-facing ask request."""

    query: str = Field(..., min_length=1, max_length=2000)


class AskResponse(BaseModel):
    """User-facing ask response.

    Only the final answer is returned to the frontend. `run_id` is included for
    backend troubleshooting but does not expose retrieval or context details.
    """

    answer: str
    run_id: Optional[str] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
