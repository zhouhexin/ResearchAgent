"""Runtime settings for the public QA API."""

from __future__ import annotations

import os
from pathlib import Path

import config


API_TOP_K = int(os.getenv("QA_API_TOP_K", str(config.TOP_K)))
API_CONTEXT_BUDGET = int(os.getenv("QA_API_CONTEXT_BUDGET", str(config.CONTEXT_BUDGET)))
API_STRATEGY = os.getenv("QA_API_STRATEGY", "baseline")
API_COMPRESSION = os.getenv("QA_API_COMPRESSION", "none")
API_COMPRESSION_STAGE = os.getenv("QA_API_COMPRESSION_STAGE", "after-allocation")
API_TEMPERATURE = float(os.getenv("QA_API_TEMPERATURE", "0.0"))
API_RUN_LABEL_PREFIX = os.getenv("QA_API_RUN_LABEL_PREFIX", "web")
API_RUNS_DIR = Path(os.getenv("QA_API_RUNS_DIR", str(config.BASE_DIR / "experiments" / "web_runs")))
