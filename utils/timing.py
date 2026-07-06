from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Any

logger = logging.getLogger("rag.timing")


@contextmanager
def log_timing(stage: str, **metadata: Any):
    """Log elapsed seconds for a pipeline stage with optional metadata."""

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        if metadata:
            details = " | ".join(f"{key}={value}" for key, value in metadata.items())
            logger.info("%s: %.3fs | %s", stage, elapsed, details)
        else:
            logger.info("%s: %.3fs", stage, elapsed)


def log_pipeline_summary(provider: str, model: str, stages: dict[str, float]) -> None:
    """Log a single end-to-end timing breakdown for one QA request."""

    total = sum(stages.values())
    breakdown = ", ".join(f"{name}={seconds:.3f}s" for name, seconds in stages.items())
    logger.info(
        "qa_pipeline_summary | provider=%s | model=%s | total=%.3fs | %s",
        provider,
        model,
        total,
        breakdown,
    )
