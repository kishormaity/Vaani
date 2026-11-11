"""Common types, protocols, and exceptions for LLM clients."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Mapping, Optional, Protocol, Tuple, runtime_checkable

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when an API rate limit or daily quota is exceeded (HTTP 429)."""

    def __init__(
        self,
        message: str,
        retry_after: float = 60.0,
        status_code: int = 429,
        limit_type: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.retry_after = retry_after
        self.status_code = status_code
        self.limit_type = limit_type

    def __repr__(self) -> str:
        return (
            f"RateLimitError(message={self.message!r}, "
            f"retry_after={self.retry_after}, status_code={self.status_code}, "
            f"limit_type={self.limit_type!r})"
        )


@dataclass
class RateLimitTelemetry:
    """Captured HTTP response rate-limit telemetry from LLM provider."""

    limit_requests_rpm: Optional[int] = None
    remaining_requests: Optional[int] = None
    reset_requests_duration: Optional[str] = None
    limit_tokens_tpm: Optional[int] = None
    remaining_tokens: Optional[int] = None
    reset_tokens_duration: Optional[str] = None
    retry_after: Optional[float] = None
    raw_headers: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_headers(cls, headers: Optional[Mapping[str, str]]) -> RateLimitTelemetry:
        """Parse rate limit information from case-insensitive HTTP headers."""
        if not headers:
            return cls()

        h = {k.lower(): str(v) for k, v in headers.items()}

        def parse_int(key: str) -> Optional[int]:
            val = h.get(key)
            if val is not None:
                try:
                    return int(val.strip())
                except ValueError:
                    return None
            return None

        def parse_float(key: str) -> Optional[float]:
            val = h.get(key)
            if val is not None:
                try:
                    return float(val.strip())
                except ValueError:
                    return None
            return None

        return cls(
            limit_requests_rpm=parse_int("x-ratelimit-limit-requests"),
            remaining_requests=parse_int("x-ratelimit-remaining-requests"),
            reset_requests_duration=h.get("x-ratelimit-reset-requests"),
            limit_tokens_tpm=parse_int("x-ratelimit-limit-tokens"),
            remaining_tokens=parse_int("x-ratelimit-remaining-tokens"),
            reset_tokens_duration=h.get("x-ratelimit-reset-tokens"),
            retry_after=parse_float("retry-after"),
            raw_headers=dict(h),
        )


@dataclass
class CompletionResult:
    """Summary of a completed streaming LLM generation."""

    content: str
    start_time: float
    first_token_ms: float
    total_ms: float
    tokens_in: int
    tokens_out: int
    telemetry: RateLimitTelemetry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "start_time": self.start_time,
            "first_token_ms": round(self.first_token_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "telemetry": asdict(self.telemetry),
        }


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol satisfied by both real Groq client and simulated stand-in service."""

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> AsyncGenerator[Tuple[str, Optional[CompletionResult]], None]:
        """Stream tokens as (token_text, None), ending with ('', CompletionResult)."""
        ...

    async def complete_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> CompletionResult:
        """Fully consume the stream and return the final CompletionResult."""
        ...

