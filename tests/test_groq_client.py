"""Unit tests for Groq LLM client and telemetry tracking with mocked streaming responses."""

import asyncio
import time
import unittest
from unittest.mock import AsyncMock, MagicMock

from llm.groq_client import CompletionResult, GroqLLMClient, RateLimitTelemetry


class MockDelta:
    def __init__(self, content: str) -> None:
        self.content = content


class MockChoice:
    def __init__(self, content: str) -> None:
        self.delta = MockDelta(content)


class MockUsage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class MockStreamChunk:
    def __init__(self, content: str = "", usage: MockUsage = None) -> None:
        self.choices = [MockChoice(content)] if content else []
        self.usage = usage


class TestGroqClientMocked(unittest.IsolatedAsyncioTestCase):
    """Test suite verifying streaming latency (TTFT), token counts, and telemetry without live network calls."""

    def test_rate_limit_telemetry_parsing(self) -> None:
        """Verify parsing of rate-limit headers into structured telemetry."""
        headers = {
            "x-ratelimit-limit-requests": "30",
            "x-ratelimit-remaining-requests": "29",
            "x-ratelimit-reset-requests": "2.1s",
            "x-ratelimit-limit-tokens": "7000",
            "x-ratelimit-remaining-tokens": "6920",
            "retry-after": "5.5",
        }
        telem = RateLimitTelemetry.from_headers(headers)
        self.assertEqual(telem.limit_requests_rpm, 30)
        self.assertEqual(telem.remaining_requests, 29)
        self.assertEqual(telem.reset_requests_duration, "2.1s")
        self.assertEqual(telem.limit_tokens_tpm, 7000)
        self.assertEqual(telem.remaining_tokens, 6920)
        self.assertEqual(telem.retry_after, 5.5)

    async def test_streaming_ttft_and_latency_recording(self) -> None:
        """Verify that TTFT and total latency are accurately measured during token streaming."""
        # Setup mock chunks with controlled delays
        async def mock_chunk_stream():
            await asyncio.sleep(0.015)  # Simulate network latency before first token (~15ms)
            yield MockStreamChunk(content="Hello")
            await asyncio.sleep(0.010)
            yield MockStreamChunk(content=" world")
            yield MockStreamChunk(content="!", usage=MockUsage(prompt_tokens=8, completion_tokens=3))

        mock_raw_response = MagicMock()
        mock_raw_response.headers = {
            "x-ratelimit-limit-requests": "30",
            "x-ratelimit-remaining-requests": "25",
            "x-ratelimit-limit-tokens": "6000",
            "x-ratelimit-remaining-tokens": "5950",
        }
        mock_raw_response.parse.return_value = mock_chunk_stream()

        # Setup mock AsyncGroq client
        mock_client = MagicMock()
        mock_client.chat.completions.with_raw_response.create = AsyncMock(return_value=mock_raw_response)

        llm_client = GroqLLMClient(api_key="mock_key", async_client=mock_client)

        result: CompletionResult = await llm_client.complete_stream(
            messages=[{"role": "user", "content": "Say hello"}]
        )

        # Assert content assembly
        self.assertEqual(result.content, "Hello world!")

        # Assert TTFT and total latency mathematics
        self.assertGreater(result.first_token_ms, 10.0, "TTFT must be at least 10ms")
        self.assertGreaterEqual(result.total_ms, result.first_token_ms, "Total latency must exceed TTFT")
        self.assertEqual(result.tokens_in, 8)
        self.assertEqual(result.tokens_out, 3)

        # Assert telemetry
        self.assertEqual(result.telemetry.limit_requests_rpm, 30)
        self.assertEqual(result.telemetry.remaining_requests, 25)
        self.assertEqual(result.telemetry.limit_tokens_tpm, 6000)

    def test_missing_api_key_raises_informative_error(self) -> None:
        """Verify that attempting to access client without an API key raises ValueError."""
        client = GroqLLMClient(api_key=None, async_client=None)
        # Clear any env var temporarily
        client.api_key = None
        with self.assertRaises(ValueError) as ctx:
            _ = client.client
        self.assertIn("GROQ_API_KEY is not set", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()

