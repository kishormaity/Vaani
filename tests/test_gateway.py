"""Comprehensive unit tests for Phase 4: The LLM Gateway (gateway/).

Tests cover:
1. Concurrency limit (at most 15 concurrent in-flight requests).
2. Bounded queue overflow (rejection when queue capacity is exceeded).
3. Token bucket rate limiting (pausing/throttling on 30 RPM or 8000 TPM).
4. Daily budget cutoff (RPD and TPD exhaustion refusals).
5. Exponential backoff and jitter on HTTP 429 with dynamic retry-after parsing.
6. Non-transient error bypass (no retries on 4xx permanent errors).
7. Integration with RAGOrchestrator / RelayPipelineOrchestrator.
"""

from __future__ import annotations

import asyncio
import sys
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gateway import (
    DailyBudgetExceededError,
    LLMGateway,
    LLMGatewayConfig,
    QueueOverflowError,
    RetryPolicy,
    RetryPolicyConfig,
    TokenBucketRateLimiter,
)
from llm.base import CompletionResult, RateLimitError, RateLimitTelemetry
from llm.standin_service import StandinConfig, StandinLLMService
from pipeline.orchestrator import RAGOrchestrator, RelayPipelineOrchestrator
from retrieval.retriever import RetrievedChunk


def make_mock_completion(content: str = "Test answer [arXiv:1706.03762v7, p.4].") -> CompletionResult:
    return CompletionResult(
        content=content,
        start_time=time.time(),
        first_token_ms=50.0,
        total_ms=100.0,
        tokens_in=50,
        tokens_out=25,
        telemetry=RateLimitTelemetry(),
    )


class TestTokenBucketRateLimiter(unittest.IsolatedAsyncioTestCase):
    """Unit tests for TokenBucketRateLimiter rolling 60s windows."""

    def test_acquire_capacity_and_rpm_throttling(self) -> None:
        """Verify acquire returns 0.0 when capacity is available and >0 when RPM is exceeded."""
        limiter = TokenBucketRateLimiter(rpm_limit=3, tpm_limit=5000, window_seconds=10.0)

        # First 3 requests should be accepted immediately
        for _ in range(3):
            delay = limiter.acquire(estimated_tokens=50)
            self.assertEqual(delay, 0.0)

        # 4th request should require a sleep delay
        delay_4 = limiter.acquire(estimated_tokens=50)
        self.assertGreater(delay_4, 0.0)

    def test_acquire_tpm_throttling(self) -> None:
        """Verify acquire returns >0 when TPM limit would be breached."""
        limiter = TokenBucketRateLimiter(rpm_limit=10, tpm_limit=500, window_seconds=10.0)

        # 1st request with 400 tokens -> ok
        self.assertEqual(limiter.acquire(estimated_tokens=400), 0.0)

        # 2nd request with 200 tokens -> breaches 500 TPM limit
        delay = limiter.acquire(estimated_tokens=200)
        self.assertGreater(delay, 0.0)

    async def test_wait_and_acquire_recovers_after_window(self) -> None:
        """Verify wait_and_acquire pauses and acquires once rolling window elapses."""
        # Use short 0.1s window for quick test
        limiter = TokenBucketRateLimiter(rpm_limit=2, tpm_limit=1000, window_seconds=0.1)

        self.assertEqual(limiter.acquire(10), 0.0)
        self.assertEqual(limiter.acquire(10), 0.0)

        # Exceeded: wait_and_acquire should wait ~0.1s and then succeed
        t0 = time.perf_counter()
        waited = await limiter.wait_and_acquire(10, max_wait=2.0)
        t_elapsed = time.perf_counter() - t0

        self.assertGreater(t_elapsed, 0.05)
        self.assertGreater(waited, 0.0)

    def test_record_usage_reconciles_reservation(self) -> None:
        """Verify record_usage updates the actual consumed tokens in history."""
        limiter = TokenBucketRateLimiter(rpm_limit=10, tpm_limit=5000)
        limiter.acquire(estimated_tokens=150)
        stats_before = limiter.get_window_stats()
        self.assertEqual(stats_before["used_tokens"], 150)

        limiter.record_usage(actual_tokens=85)
        stats_after = limiter.get_window_stats()
        self.assertEqual(stats_after["used_tokens"], 85)


class TestRetryPolicy(unittest.IsolatedAsyncioTestCase):
    """Unit tests for RetryPolicy exponential backoff and jitter."""

    async def test_retry_on_429_with_retry_after(self) -> None:
        """Verify retry policy honors 429 retry-after and succeeds on subsequent attempt."""
        policy = RetryPolicy(RetryPolicyConfig(base_backoff=0.05, jitter=0.01))

        mock_call = AsyncMock()
        rate_err = RateLimitError("Rate limit reached", retry_after=0.05, status_code=429)
        mock_call.side_effect = [rate_err, "success_result"]

        res, retries, delay = await policy.execute(mock_call)

        self.assertEqual(res, "success_result")
        self.assertEqual(retries, 1)
        self.assertGreater(delay, 0.04)
        self.assertEqual(mock_call.call_count, 2)

    async def test_non_transient_error_fails_immediately(self) -> None:
        """Verify 4xx non-transient errors (like 404, 400) are never retried."""
        policy = RetryPolicy(RetryPolicyConfig(base_backoff=0.01))

        class Client404Error(Exception):
            status_code = 404

        mock_call = AsyncMock(side_effect=Client404Error("Model not found"))

        with self.assertRaises(Client404Error):
            await policy.execute(mock_call)

        self.assertEqual(mock_call.call_count, 1)


class TestLLMGateway(unittest.IsolatedAsyncioTestCase):
    """Unit tests for the full LLMGateway concurrency, queuing, rate limiting, and quotas."""

    async def test_concurrency_limit_capped_at_15(self) -> None:
        """Ensure no more than 15 requests run in flight simultaneously."""
        in_flight = 0
        max_seen = 0
        lock = asyncio.Lock()

        async def _mock_stream(messages, max_tokens=150, temperature=0.0):
            nonlocal in_flight, max_seen
            async with lock:
                in_flight += 1
                if in_flight > max_seen:
                    max_seen = in_flight
            await asyncio.sleep(0.05)
            async with lock:
                in_flight -= 1
            return make_mock_completion()

        mock_client = MagicMock()
        mock_client.complete_stream = AsyncMock(side_effect=_mock_stream)
        mock_client.source = "mock-client"

        config = LLMGatewayConfig(max_concurrency=15, max_queue_size=50, rpm_limit=1000)
        gateway = LLMGateway(client=mock_client, config=config)

        # Launch 25 simultaneous workers
        tasks = [gateway.complete_stream([{"role": "user", "content": "test"}]) for _ in range(25)]
        results = await asyncio.gather(*tasks)

        self.assertEqual(len(results), 25)
        self.assertEqual(max_seen, 15)
        self.assertLessEqual(max_seen, 15)

    async def test_queue_overflow_rejection(self) -> None:
        """Ensure requests beyond queue capacity (maxsize=3) reject with QueueOverflowError."""
        mock_client = MagicMock()
        mock_client.complete_stream = AsyncMock(return_value=make_mock_completion())

        config = LLMGatewayConfig(max_concurrency=1, max_queue_size=3, rpm_limit=1000)
        gateway = LLMGateway(client=mock_client, config=config)

        # Pre-fill the bounded queue to maximum capacity
        for i in range(config.max_queue_size):
            gateway.queue.put_nowait(f"prefill_{i}")

        # Any subsequent request must immediately raise QueueOverflowError
        with self.assertRaises(QueueOverflowError) as ctx:
            await gateway.complete_stream([{"role": "user", "content": "overflow"}])

        self.assertEqual(ctx.exception.status_code, 503)
        self.assertIn("capacity of 3 exceeded", str(ctx.exception))

    async def test_daily_request_budget_cutoff(self) -> None:
        """Ensure requests reject cleanly with DailyBudgetExceededError when RPD is exhausted."""
        mock_client = MagicMock()
        mock_client.complete_stream = AsyncMock(return_value=make_mock_completion())

        config = LLMGatewayConfig(rpd_limit=2, rpm_limit=1000)
        gateway = LLMGateway(client=mock_client, config=config)

        # 2 requests succeed
        await gateway.complete_stream([{"role": "user", "content": "1"}])
        await gateway.complete_stream([{"role": "user", "content": "2"}])
        self.assertEqual(gateway.daily_requests, 2)

        # 3rd request rejected
        with self.assertRaises(DailyBudgetExceededError) as ctx:
            await gateway.complete_stream([{"role": "user", "content": "3"}])

        self.assertEqual(ctx.exception.status_code, 429)
        self.assertEqual(ctx.exception.limit_type, "RPD")

    async def test_daily_token_budget_cutoff(self) -> None:
        """Ensure requests reject cleanly when daily token limit (TPD) is reached."""
        mock_client = MagicMock()
        mock_client.complete_stream = AsyncMock(return_value=make_mock_completion())

        config = LLMGatewayConfig(tpd_limit=100, rpm_limit=1000)
        gateway = LLMGateway(client=mock_client, config=config)

        # Artificially set daily tokens at limit
        gateway._daily_tokens = 100

        with self.assertRaises(DailyBudgetExceededError) as ctx:
            await gateway.complete_stream([{"role": "user", "content": "test"}])

        self.assertEqual(ctx.exception.limit_type, "TPD")

    async def test_backoff_on_429_then_success(self) -> None:
        """Verify backoff triggers on 429 and succeeds on attempt 1, logging retries."""
        rate_err = RateLimitError("Rate limit exceeded", retry_after=0.05, status_code=429)
        good_res = make_mock_completion()

        mock_client = MagicMock()
        mock_client.complete_stream = AsyncMock(side_effect=[rate_err, good_res])
        mock_client.source = "mock-groq"

        config = LLMGatewayConfig(base_backoff=0.05, jitter=0.01)
        gateway = LLMGateway(client=mock_client, config=config)

        result = await gateway.complete_stream([{"role": "user", "content": "test"}])

        self.assertIsNotNone(result)
        self.assertEqual(gateway.get_metrics()["total_retries"], 1)
        self.assertEqual(result.telemetry.raw_headers["x-gateway-retries"], "1")

    async def test_orchestrator_integration_with_gateway(self) -> None:
        """Verify RelayPipelineOrchestrator integrates seamlessly with LLMGateway."""
        standin = StandinLLMService(
            config=StandinConfig(
                base_ttft_ms=10.0,
                ms_per_token=1.0,
                failure_rate=0.0,
                simulate_latency=True,
            )
        )
        gateway = LLMGateway(client=standin, config=LLMGatewayConfig(rpm_limit=100))

        mock_chunks = [
            RetrievedChunk(
                chunk_id="c1",
                paper_id="1706.03762",
                paper_title="Attention Is All You Need",
                version=7,
                page_numbers=[4, 5],
                text="The multi-head attention mechanism computes representations in parallel.",
                score=0.88,
                citation="[arXiv:1706.03762v7, pp.4, 5]",
            )
        ]
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_chunks

        orchestrator = RelayPipelineOrchestrator(
            retriever=mock_retriever,
            llm_client=gateway,
        )

        result = await orchestrator.execute("What is the transformer multi-head attention?")

        self.assertFalse(result.is_refusal)
        self.assertEqual(result.loop_count, 0)
        self.assertIn("[arXiv:1706.03762", result.answer)
        self.assertEqual(gateway.get_metrics()["total_completed"], 1)


if __name__ == "__main__":
    unittest.main()
