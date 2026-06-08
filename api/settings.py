"""Runtime settings for the public QA API."""

from __future__ import annotations

import os

import config


API_TOP_K = int(os.getenv("QA_API_TOP_K", str(config.TOP_K)))
API_CONTEXT_BUDGET = int(os.getenv("QA_API_CONTEXT_BUDGET", str(config.CONTEXT_BUDGET)))
API_STRATEGY = os.getenv("QA_API_STRATEGY", "baseline")
API_COMPRESSION = os.getenv("QA_API_COMPRESSION", "none")
API_COMPRESSION_STAGE = os.getenv("QA_API_COMPRESSION_STAGE", "after-allocation")
API_RUN_LABEL_PREFIX = os.getenv("QA_API_RUN_LABEL_PREFIX", "web")
