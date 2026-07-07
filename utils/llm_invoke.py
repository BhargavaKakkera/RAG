from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("rag.llm")

FALLBACK_GROQ_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"


@dataclass(frozen=True)
class InvokeResult:
    content: Any


def _is_timeout_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "timeout" in msg or "timed out" in msg or "deadline" in msg



def _is_http_429(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "429" in msg or "rate limit" in msg


def _is_http_503(exc: BaseException) -> bool:
    msg = str(exc).lower()
    return "503" in msg or "service unavailable" in msg


def _should_retry(exc: BaseException) -> bool:
    return _is_timeout_error(exc) or _is_http_429(exc) or _is_http_503(exc)


def _invoke_with_model(
    provider: str,
    model: str,
    chain_name: str,
    model_setter: Callable[[str], Any],
    chain_callable: Callable[[], Any],
    inputs: dict[str, Any],
) -> tuple[Any, float]:
    start = time.perf_counter()
    model_setter(model)
    try:
        result = chain_callable()
    finally:
        elapsed = time.perf_counter() - start

    return result, elapsed


def invoke_llm(
    *,
    provider: str,
    original_model: str,
    chain_name: str,
    fallback_model: str = FALLBACK_GROQ_MODEL,
    max_retries: int = 3,
    retry_backoff_seconds: tuple[int, int, int] = (1, 2, 4),
    model_setter: Callable[[str], Any],
    chain_callable: Callable[[], Any],
    inputs: dict[str, Any],
) -> Any:
    """Invoke LLM with retry on 429/503/timeout and fallback to scout-17b.

    Notes:
    - Retries use the same model.
    - Fallback is only used for the current request.
    """

    retry_count = 0
    fallback_used = False

    def log_request(model: str, retry_count: int, fallback_used: bool, elapsed: float) -> None:
        logger.info(
            "llm_request | provider=%s | model=%s | chain=%s | retry_count=%s | fallback_used=%s | elapsed_s=%.3f",
            provider,
            model,
            chain_name,
            retry_count,
            fallback_used,
            elapsed,
        )

    # First try original model with up to max_retries retries (total attempts = max_retries+1).
    current_model = original_model
    while True:
        try:
            result, elapsed = _invoke_with_model(
                provider=provider,
                model=current_model,
                chain_name=chain_name,
                model_setter=model_setter,
                chain_callable=chain_callable,
                inputs=inputs,
            )
            log_request(
                model=current_model,
                retry_count=retry_count,
                fallback_used=fallback_used,
                elapsed=elapsed,
            )
            return result
        except Exception as exc:  # noqa: BLE001
            if _should_retry(exc) and retry_count < max_retries:
                backoff = retry_backoff_seconds[retry_count]
                retry_count += 1
                logger.warning(
                    "llm_retry | provider=%s | chain=%s | model=%s | retry=%s/%s | backoff_s=%s | error=%s",
                    provider,
                    chain_name,
                    current_model,
                    retry_count,
                    max_retries,
                    backoff,
                    exc,
                )
                time.sleep(backoff)
                continue

            # If we get here: either not retryable, or exhausted retries.
            # Apply fallback only if we haven't already fallen back.
            if not fallback_used:
                fallback_used = True
                fallback_model_effective = fallback_model
                if fallback_model_effective != current_model:
                    fallback_reason = (
                        "rate-limited/service-unavailable/timeout: switching to higher TPM fallback"
                    )
                    logger.warning(
                        "llm_fallback | provider=%s | chain=%s | original_model=%s | retry_count=%s | fallback_model=%s | reason=%s",
                        provider,
                        chain_name,
                        original_model,
                        retry_count,
                        fallback_model_effective,
                        fallback_reason,
                    )
                    current_model = fallback_model_effective
                    # reset retry counter for the fallback request (still part of the same request)
                    retry_count = 0
                    # Retry the call immediately with the fallback model.
                    continue
                # If fallback_model equals original, nothing else to do; re-raise.
            raise


