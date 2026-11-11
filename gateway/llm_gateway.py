"""LLM Gateway providing concurrency control, bounded request queuing,
token bucket rate limiting, daily quota management, and resilient retries.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from gateway.retry_policy import RetryPolicy, RetryPolicyConfig
from gateway.token_bucket import TokenBucketRateLimiter
from llm.base import CompletionResult, LLMClientProtocol, RateLimitError, RateLimitTelemetry
from llm.standin_service import estimate_tokens

logger = logging.getLogger(__name__)


class QueueOverflowError(Exception):
    """Raised when the gateway request queue exceeds capacity (HTTP 503)."""

    def __init__(self, message: str = "Gateway request queue overflow: capacity exceeded.") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = 503


class DailyBudgetExceededError(RateLimitError):
    """Raised when the daily request (RPD) or token (TPD) quota is exhausted."""

    def __init__(self, message: str, limit_type: str = "RPD") -> None:
        super().__init__(message=message, retry_after=86400.0, status_code=429, limit_type=limit_type)


@dataclass
class LLMGatewayConfig:
    """Configuration for LLMGateway."""

    # Concurrency control
    max_concurrency: int = 15

    # Bounded queue
    max_queue_size: int = 50

    # Rate limiting (RPM / TPM)
    rpm_limit: int = 30
    tpm_limit: int = 8000
    window_seconds: float = 60.0

    # Daily budget limits (RPD / TPD)
    rpd_limit: int = 1000
    tpd_limit: int = 200000

    # Retry policy
    max_retries: int = 3
    base_backoff: float = 2.0
    max_backoff: float = 60.0
    jitter: float = 0.5


class LLMGateway:
    """Wraps an LLM client with queuing, rate limiting, and backoff."""

    def __init__(
        self,
        client: Optional[LLMClientProtocol] = None,
        config: Optional[LLMGatewayConfig] = None,
    ) -> None:
        self.config = config or LLMGatewayConfig()

        if client is None:
            from llm import GroqLLMClient

            self.client: LLMClientProtocol = GroqLLMClient()
        else:
            self.client = client

        # Concurrency & queue
        self.semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self.queue: asyncio.Queue[str] = asyncio.Queue(maxsize=self.config.max_queue_size)
        self._peak_queue_length: int = 0
        self._active_in_flight: int = 0

        # Rate limiter
        self.rate_limiter = TokenBucketRateLimiter(
            rpm_limit=self.config.rpm_limit,
            tpm_limit=self.config.tpm_limit,
            window_seconds=self.config.window_seconds,
        )

        # Retry policy
        self.retry_policy = RetryPolicy(
            RetryPolicyConfig(
                max_retries=self.config.max_retries,
                base_backoff=self.config.base_backoff,
                max_backoff=self.config.max_backoff,
                jitter=self.config.jitter,
            )
        )

        # Daily budget tracking
        self._daily_requests: int = 0
        self._daily_tokens: int = 0

        # Telemetry metrics
        self._total_completed: int = 0
        self._total_rejected: int = 0
        self._total_retries: int = 0
        self._total_throttled: int = 0

        self._state_lock = asyncio.Lock()

    @property
    def source(self) -> str:
        return getattr(self.client, "source", "unknown")

    @property
    def model(self) -> str:
        return getattr(self.client, "model", "unknown")

    @property
    def current_queue_depth(self) -> int:
        return self.queue.qsize()

    @property
    def peak_queue_length(self) -> int:
        return self._peak_queue_length

    @property
    def active_in_flight(self) -> int:
        return self._active_in_flight

    @property
    def daily_requests(self) -> int:
        return self._daily_requests

    @property
    def daily_tokens(self) -> int:
        return self._daily_tokens

    def _check_daily_budget(self) -> None:
        """Check if daily requests or tokens quota is exhausted."""
        if self._daily_requests >= self.config.rpd_limit:
            raise DailyBudgetExceededError(
                f"Daily request budget exhausted: limit of {self.config.rpd_limit} RPD reached.",
                limit_type="RPD",
            )
        if self._daily_tokens >= self.config.tpd_limit:
            raise DailyBudgetExceededError(
                f"Daily token budget exhausted: limit of {self.config.tpd_limit} TPD reached.",
                limit_type="TPD",
            )

    async def complete_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> CompletionResult:
        """Process completion request through gateway pipeline."""
        req_id = str(uuid.uuid4())

        # 1. Enforce Daily Budget
        async with self._state_lock:
            self._check_daily_budget()

        # 2. Check Queue Capacity
        try:
            self.queue.put_nowait(req_id)
        except asyncio.QueueFull:
            async with self._state_lock:
                self._total_rejected += 1
            logger.warning(
                "Gateway request rejected: queue capacity (%d) exceeded.",
                self.config.max_queue_size,
            )
            raise QueueOverflowError(
                f"Gateway request queue overflow: capacity of {self.config.max_queue_size} exceeded."
            )

        async with self._state_lock:
            if self.queue.qsize() > self._peak_queue_length:
                self._peak_queue_length = self.queue.qsize()

        # 3. Wait for Concurrency Semaphore
        t_enter_queue = time.perf_counter()
        async with self.semaphore:
            queuing_delay_ms = (time.perf_counter() - t_enter_queue) * 1000.0

            # Dequeue ticket
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except asyncio.QueueEmpty:
                pass

            async with self._state_lock:
                self._active_in_flight += 1

            try:
                # 4. Token Bucket Rate Limiting
                est_tokens = sum(estimate_tokens(m.get("content", "")) for m in messages) + max_tokens
                wait_sec = await self.rate_limiter.wait_and_acquire(estimated_tokens=est_tokens)
                if wait_sec > 0.05:
                    async with self._state_lock:
                        self._total_throttled += 1

                # 5. Execute with Exponential Backoff Retries
                async def _call():
                    return await self.client.complete_stream(
                        messages=messages,
                        max_tokens=max_tokens,
                        temperature=temperature,
                    )

                result, retries, retry_delay_sec = await self.retry_policy.execute(_call)

                # 6. Reconcile Actual Usage & Update State
                actual_tokens = result.tokens_in + result.tokens_out
                self.rate_limiter.record_usage(actual_tokens)

                async with self._state_lock:
                    self._daily_requests += 1
                    self._daily_tokens += actual_tokens
                    self._total_completed += 1
                    self._total_retries += retries

                # Annotate result telemetry with gateway metrics
                result.telemetry.raw_headers["x-gateway-queuing-delay-ms"] = str(round(queuing_delay_ms, 2))
                result.telemetry.raw_headers["x-gateway-retries"] = str(retries)
                result.telemetry.raw_headers["x-gateway-peak-queue"] = str(self._peak_queue_length)

                return result

            finally:
                async with self._state_lock:
                    self._active_in_flight = max(0, self._active_in_flight - 1)

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> AsyncGenerator[Tuple[str, Optional[CompletionResult]], None]:
        """Process streaming chat completion through gateway."""
        result = await self.complete_stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        yield (result.content, None)
        yield ("", result)

    def get_metrics(self) -> Dict[str, Any]:
        """Return gateway status and cumulative telemetry for Table 2b."""
        bucket_stats = self.rate_limiter.get_window_stats()
        return {
            "current_queue_depth": self.current_queue_depth,
            "peak_queue_length": self.peak_queue_length,
            "active_in_flight": self.active_in_flight,
            "daily_requests": self.daily_requests,
            "daily_tokens": self.daily_tokens,
            "rpd_limit": self.config.rpd_limit,
            "tpd_limit": self.config.tpd_limit,
            "total_completed": self._total_completed,
            "total_rejected": self._total_rejected,
            "total_retries": self._total_retries,
            "total_throttled": self._total_throttled,
            "rolling_60s": bucket_stats,
        }

    def reset_metrics(self) -> None:
        """Reset state and counters (for tests)."""
        self._daily_requests = 0
        self._daily_tokens = 0
        self._total_completed = 0
        self._total_rejected = 0
        self._total_retries = 0
        self._total_throttled = 0
        self._peak_queue_length = 0
        self._active_in_flight = 0
        self.rate_limiter.reset()
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
                self.queue.task_done()
            except Exception:
                break

