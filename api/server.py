"""FastAPI server for the laboratory knowledge QA frontend."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from api.schemas import AskRequest, AskResponse, FeedbackRequest, FeedbackResponse, HealthResponse
from api.services.feedback_service import record_feedback
from api.services.paper_service import build_or_update_paper_index, find_paper_path
from api.services.qa_service import ask_public_question


app = FastAPI(title="ResearchAgent Knowledge QA API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    """Refresh local PDF title cache when the API process starts."""
    try:
        build_or_update_paper_index()
    except Exception:
        # Paper links are an optional frontend enhancement; QA must still start.
        pass


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Return API health status."""
    return HealthResponse(status="ok")


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    """Ask the knowledge base and return the final answer only."""
    query = request.query.strip()
    if not query:
        return AskResponse(answer="", run_id=None, error="问题不能为空")
    try:
        result = ask_public_question(query)
    except Exception as exc:  # noqa: BLE001 - API boundary returns safe error text.
        return AskResponse(answer="", run_id=None, error=str(exc))
    return AskResponse(**result)


@app.post("/feedback", response_model=FeedbackResponse)
def feedback(request: FeedbackRequest) -> FeedbackResponse:
    """Persist user feedback for a displayed answer."""
    try:
        record_feedback(
            run_id=request.run_id,
            query=request.query.strip(),
            answer=request.answer.strip(),
            rating=request.rating,
        )
    except Exception as exc:  # noqa: BLE001 - API boundary returns safe error text.
        return FeedbackResponse(ok=False, error=str(exc))
    return FeedbackResponse(ok=True)


@app.get("/papers/file/{paper_id}")
def paper_file(paper_id: str, download: bool = Query(default=False)) -> FileResponse:
    """Preview or download a local PDF by public paper id."""
    path = find_paper_path(paper_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Paper not found")
    return FileResponse(
        path,
        media_type="application/pdf",
        filename=path.name if download else None,
    )
