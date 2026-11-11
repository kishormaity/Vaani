"""Unit tests for the stand-in model service (StandinLLMService).

Tests cover:
1. TTFT lognormal distribution calibration (median ~145 ms, p95 ~220 ms).
2. Streaming token generation and protocol conformity.
3. RPM and TPM rate limit tripping with 429 RateLimitError.
4. Concurrency overload degradation scaling.
5. Daily quota exhaustion (RPD and TPD).
6. Integration with RAGPipeline.
"""

from __future__ import annotations

import asyncio
import statistics
import unittest
from unittest.mock import MagicMock

from llm.base import CompletionResult, LLMClientProtocol, RateLimitError
from llm.standin_service import StandinConfig, StandinLLMService
from rag.pipeline import RAGPipeline
from retrieval.retriever import RetrievedChunk


class TestStandinLLMService(unittest.IsolatedAsyncioTestCase):
    """Test suite for StandinLLMService calibrated behavior and limits."""

    def test_protocol_conformance(self) -> None:
        """Verify that StandinLLMService satisfies LLMClientProtocol."""
        service = StandinLLMService()
        self.assertIsInstance(service, LLMClientProtocol)

    def test_ttft_calibration_distribution(self) -> None:
        """Verify that TTFT sampling follows lognormal distribution with median ~613ms and p95 ~2565ms."""
        config = StandinConfig(
            base_ttft_ms=613.0,
            p95_ttft_ms=2565.0,
            seed=42,
        )
        service = StandinLLMService(config=config)

        # Sample 5,000 TTFT measurements
        samples = [service.sample_ttft() for _ in range(5000)]
        samples.sort()

        median_ttft = statistics.median(samples)
        p95_index = int(0.95 * len(samples))
        p95_ttft = samples[p95_index]

        # Assert median is close to 613 ms (within ±25 ms)
        self.assertAlmostEqual(median_ttft, 613.0, delta=25.0)

        # Assert p95 is close to 2565 ms (within ±75 ms)
        self.assertAlmostEqual(p95_ttft, 2565.0, delta=75.0)

        # Assert all samples are positive non-zero
        self.assertGreater(min(samples), 10.0)

    async def test_streaming_chat_yields_expected_protocol(self) -> None:
        """Verify stream_chat yields (token, None) chunks and final ('', CompletionResult)."""
        config = StandinConfig(
            base_ttft_ms=10.0,
            ms_per_token=1.0,
            failure_rate=0.0,
            simulate_latency=True,
            seed=123,
        )
        service = StandinLLMService(config=config)

        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": "Context: [arXiv:1706.03762v7, p.5] Explain self-attention.",
            },
        ]

        streamed_tokens = []
        final_result = None

        async for chunk, res in service.stream_chat(messages=messages, max_tokens=30):
            if chunk:
                streamed_tokens.append(chunk)
            if res is not None:
                final_result = res

        self.assertIsNotNone(final_result)
        self.assertIsInstance(final_result, CompletionResult)
        self.assertTrue(len(streamed_tokens) > 0)
        self.assertEqual("".join(streamed_tokens), final_result.content)
        self.assertIn("[arXiv:1706.03762v7, p.5]", final_result.content)
        self.assertGreater(final_result.first_token_ms, 0.0)
        self.assertGreaterEqual(final_result.total_ms, final_result.first_token_ms)
        self.assertGreater(final_result.tokens_in, 0)
        self.assertGreater(final_result.tokens_out, 0)

        # Verify telemetry structure (default RPM is 1000)
        self.assertEqual(final_result.telemetry.limit_requests_rpm, 1000)
        self.assertEqual(final_result.telemetry.remaining_requests, 999)
        self.assertIn("x-ratelimit-limit-requests", final_result.telemetry.raw_headers)

    async def test_rpm_limit_tripping(self) -> None:
        """Verify that exceeding rpm_limit raises RateLimitError with status_code 429 and retry_after."""
        config = StandinConfig(
            rpm_limit=5,
            simulate_latency=False,
            failure_rate=0.0,
        )
        service = StandinLLMService(config=config)

        messages = [{"role": "user", "content": "Test prompt"}]

        # First 5 requests should succeed
        for i in range(5):
            res = await service.complete_stream(messages)
            self.assertIsNotNone(res)

        self.assertEqual(len(service._minute_window), 5)

        # 6th request within same minute must raise RateLimitError
        with self.assertRaises(RateLimitError) as ctx:
            await service.complete_stream(messages)

        exc = ctx.exception
        self.assertEqual(exc.status_code, 429)
        self.assertEqual(exc.limit_type, "RPM")
        self.assertGreater(exc.retry_after, 0.0)
        self.assertIn("RPM limit of 5 reached", exc.message)

    async def test_tpm_limit_tripping(self) -> None:
        """Verify that exceeding tpm_limit raises RateLimitError (TPM)."""
        config = StandinConfig(
            tpm_limit=50,
            simulate_latency=False,
            failure_rate=0.0,
        )
        service = StandinLLMService(config=config)

        # Prompt with ~60 tokens exceeds the 50 TPM limit immediately
        large_prompt = "word " * 60
        messages = [{"role": "user", "content": large_prompt}]

        with self.assertRaises(RateLimitError) as ctx:
            await service.complete_stream(messages)

        exc = ctx.exception
        self.assertEqual(exc.status_code, 429)
        self.assertEqual(exc.limit_type, "TPM")
        self.assertIn("TPM limit of 50 reached", exc.message)

    async def test_rpd_limit_daily_exhaustion(self) -> None:
        """Verify that exceeding daily request quota raises RateLimitError with 86400s retry_after."""
        config = StandinConfig(
            rpd_limit=3,
            simulate_latency=False,
            failure_rate=0.0,
        )
        service = StandinLLMService(config=config)
        messages = [{"role": "user", "content": "ping"}]

        for _ in range(3):
            await service.complete_stream(messages)

        self.assertEqual(service.daily_requests, 3)

        # 4th request exceeds daily quota
        with self.assertRaises(RateLimitError) as ctx:
            await service.complete_stream(messages)

        exc = ctx.exception
        self.assertEqual(exc.status_code, 429)
        self.assertEqual(exc.limit_type, "RPD")
        self.assertEqual(exc.retry_after, 86400.0)
        self.assertIn("RPD limit of 3 reached", exc.message)

    async def test_tpd_limit_daily_exhaustion(self) -> None:
        """Verify that exceeding daily token quota raises RateLimitError (TPD)."""
        config = StandinConfig(
            tpd_limit=25,
            simulate_latency=False,
            failure_rate=0.0,
        )
        service = StandinLLMService(config=config)
        messages = [{"role": "user", "content": "short prompt"}]

        # First request consumes tokens
        await service.complete_stream(messages)
        self.assertGreater(service.daily_tokens, 0)

        # Manually bump daily tokens to limit
        service._daily_tokens = 30

        with self.assertRaises(RateLimitError) as ctx:
            await service.complete_stream(messages)

        exc = ctx.exception
        self.assertEqual(exc.status_code, 429)
        self.assertEqual(exc.limit_type, "TPD")
        self.assertEqual(exc.retry_after, 86400.0)

    def test_concurrency_overload_multiplier_calculation(self) -> None:
        """Verify latency degradation multiplier when active requests exceed max_concurrency."""
        config = StandinConfig(
            max_concurrency=3,
            overload_multiplier=1.5,
        )
        service = StandinLLMService(config=config)

        # Concurrency <= max_concurrency -> 1.0 multiplier
        service._active_requests = 1
        self.assertEqual(service._calculate_concurrency_multiplier(), 1.0)
        service._active_requests = 3
        self.assertEqual(service._calculate_concurrency_multiplier(), 1.0)

        # Concurrency > max_concurrency -> overload_multiplier ** (active - max_concurrency)
        service._active_requests = 4
        self.assertAlmostEqual(service._calculate_concurrency_multiplier(), 1.5)

        service._active_requests = 5
        self.assertAlmostEqual(service._calculate_concurrency_multiplier(), 1.5**2)

    async def test_concurrency_degradation_in_action(self) -> None:
        """Verify that concurrent requests dynamically scale latency when max_concurrency is exceeded."""
        config = StandinConfig(
            base_ttft_ms=20.0,
            p95_ttft_ms=20.0,  # Fixed TTFT for deterministic testing
            ms_per_token=2.0,
            max_concurrency=1,  # Overload triggers from 2nd concurrent request
            overload_multiplier=2.0,
            failure_rate=0.0,
            simulate_latency=True,
        )
        service = StandinLLMService(config=config)

        messages = [{"role": "user", "content": "Simulate concurrent load"}]

        # Run 2 requests concurrently
        res1, res2 = await asyncio.gather(
            service.complete_stream(messages),
            service.complete_stream(messages),
        )

        # At least one request was processed while active_requests > 1, so its TTFT was scaled
        max_ttft = max(res1.first_token_ms, res2.first_token_ms)
        self.assertGreater(max_ttft, 25.0)

    async def test_rag_pipeline_integration_with_standin_service(self) -> None:
        """Verify end-to-end RAG pipeline using StandinLLMService."""
        mock_chunks = [
            RetrievedChunk(
                chunk_id="chunk_42",
                paper_id="1706.03762",
                paper_title="Attention Is All You Need",
                version=7,
                page_numbers=[3, 4],
                text="The multi-head attention mechanism computes attention over projected representations.",
                score=0.92,
                citation="[arXiv:1706.03762v7, pp.3, 4]",
            )
        ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_chunks

        standin_client = StandinLLMService(
            config=StandinConfig(
                base_ttft_ms=10.0,
                ms_per_token=1.0,
                failure_rate=0.0,
                simulate_latency=True,
            )
        )

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_client=standin_client,
            setting="Setting C",
        )

        response = await pipeline.answer_query("How does multi-head attention work?")

        self.assertEqual(response.setting_name, "Setting C")
        self.assertEqual(len(response.retrieved_chunks), 1)
        self.assertIn("[arXiv:1706.03762v7, pp.3, 4]", response.citations)
        self.assertIn("attention", response.answer.lower())
        self.assertGreater(response.generation_ttft_ms, 0.0)
        self.assertGreater(response.end_to_end_ms, 0.0)


if __name__ == "__main__":
    unittest.main()

