"""Retry policy with exponential backoff, jitter, and dynamic Retry-After parsing."""

from __future__ import annotations

import asyncio
import logging
import random
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional, Tuple, TypeVar

from llm.base import RateLimitError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# HTTP status codes that represent permanent client errors (never retry)
NON_TRANSIENT_STATUS_CODES = {400, 401, 403, 404, 422}

# HTTP status codes that represent transient errors (always retry)
TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}


def extract_status_code(exc: Exception) -> Optional[int]:
    """Extract HTTP status code from an exception if present."""
    if hasattr(exc, "status_code"):
        try:
            return int(getattr(exc, "status_code"))
        except (ValueError, TypeError):
            pass
    if hasattr(exc, "response") and hasattr(exc.response, "status_code"):
        try:
            return int(getattr(exc.response, "status_code"))
        except (ValueError, TypeError):
            pass
    # Check for "Error code: 429" in error message
    msg = str(exc)
    code_match = re.search(r"Error code:\s*(\d{3})", msg)
    if code_match:
        try:
            return int(code_match.group(1))
        except ValueError:
            pass
    return None


def extract_retry_after(exc: Exception) -> Optional[float]:
    """Extract retry-after duration in seconds from exception or headers."""
    # Direct property on RateLimitError
    if hasattr(exc, "retry_after") and getattr(exc, "retry_after") is not None:
        try:
            return float(getattr(exc, "retry_after"))
        except (ValueError, TypeError):
            pass

    # Header in HTTP response
    if hasattr(exc, "response") and hasattr(exc.response, "headers"):
        headers = exc.response.headers
        val = headers.get("retry-after") or headers.get("Retry-After")
        if val is not None:
            try:
                return float(val)
            except (ValueError, TypeError):
                pass

    # Regex search in error message: "try again in 2s"
    msg = str(exc)
    match = re.search(r"try again in\s*([\d\.]+)s", msg, re.IGNORECASE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return None


def is_transient_error(exc: Exception) -> bool:
    """Determine if an error is transient and safe to retry."""
    status_code = extract_status_code(exc)
    if status_code is not None:
        if status_code in NON_TRANSIENT_STATUS_CODES:
            return False
        if status_code in TRANSIENT_STATUS_CODES:
            return True

    # Known transient exception classes
    if isinstance(exc, (RateLimitError, TimeoutError, ConnectionError, OSError)):
        return True

    exc_name = exc.__class__.__name__
    if any(k in exc_name for k in ("RateLimit", "Timeout", "Connection", "ServiceUnavailable", "InternalServer")):
        return True

    return False


@dataclass
class RetryPolicyConfig:
    """Configuration for exponential backoff and retry behavior."""

    max_retries: int = 3
    base_backoff: float = 2.0  # Groq's verified 2.0s server recovery window
    max_backoff: float = 60.0
    jitter: float = 0.5


class RetryPolicy:
    """Applies exponential backoff with full jitter and dynamic retry-after parsing."""

    def __init__(self, config: Optional[RetryPolicyConfig] = None) -> None:
        self.config = config or RetryPolicyConfig()

    def calculate_backoff(self, attempt: int, retry_after: Optional[float] = None) -> float:
        """Compute delay for a given retry attempt.

        Formula:
            delay = min(max_backoff, max(retry_after, base_backoff * (2 ** attempt))) + random.uniform(0, jitter)
        """
        r_after = float(retry_after) if retry_after is not None else 0.0
        exp_backoff = self.config.base_backoff * (2 ** attempt)
        core_delay = min(self.config.max_backoff, max(r_after, exp_backoff))
        jitter_val = random.uniform(0.0, self.config.jitter)
        return round(core_delay + jitter_val, 3)

    async def execute(
        self,
        func: Callable[[], Awaitable[T]],
    ) -> Tuple[T, int, float]:
        """Execute async function with retries on transient errors.

        Returns:
            Tuple of (result, retry_count, total_retry_delay_seconds).
        """
        attempt = 0
        total_delay = 0.0

        while True:
            try:
                result = await func()
                return result, attempt, total_delay
            except Exception as exc:
                if not is_transient_error(exc):
                    logger.error("Non-transient error encountered (%s). Not retrying.", exc)
                    raise

                if attempt >= self.config.max_retries:
                    logger.error(
                        "Max retries (%d) exhausted for transient error (%s).",
                        self.config.max_retries,
                        exc,
                    )
                    raise

                retry_after = extract_retry_after(exc)
                delay = self.calculate_backoff(attempt=attempt, retry_after=retry_after)
                logger.warning(
                    "Transient error on attempt %d (%s). Retrying in %.2fs (retry-after: %s)...",
                    attempt + 1,
                    exc,
                    delay,
                    retry_after,
                )
                total_delay += delay
                attempt += 1
                await asyncio.sleep(delay)

