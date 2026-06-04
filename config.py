"""Configuration values for ResearchAgent."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    load_dotenv = None

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INDEX_DIR = BASE_DIR / "storage"
RESULTS_PATH = BASE_DIR / "experiments" / "results.csv"
RUNS_DIR = BASE_DIR / "experiments" / "runs"

if load_dotenv is not None:
    load_dotenv(BASE_DIR / ".env")

MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_BASE_URL = os.getenv("MINIMAX_BASE_URL", "")
MINIMAX_MODEL = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/all-MiniLM-L6-v2",
)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("TOP_K", "8"))
CONTEXT_BUDGET = int(os.getenv("CONTEXT_BUDGET", "3000"))
MAX_LLM_TOKENS = int(os.getenv("MAX_LLM_TOKENS", "2048"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
LLMLINGUA2_MODEL = os.getenv(
    "LLMLINGUA2_MODEL",
    "microsoft/llmlingua-2-xlm-roberta-large-meetingbank",
)
LLMLINGUA2_RATE = float(os.getenv("LLMLINGUA2_RATE", "0.5"))

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf"}
