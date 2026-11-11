"""Groq LLM Client supporting streaming completions, latency telemetry (TTFT),
and rate limit header tracking.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from dotenv import load_dotenv

from llm.base import CompletionResult, RateLimitError, RateLimitTelemetry

logger = logging.getLogger(__name__)


class GroqLLMClient:
    """Async client for Groq LLM inference with live streaming telemetry."""

    source: str = "real"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        async_client: Optional[Any] = None,
    ) -> None:
        load_dotenv()
        self.source = "real"
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        # Default to the active model verified on your organization account
        self.model = model or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        self._async_client = async_client

    @property
    def client(self):
        """Lazily initialize or return the AsyncGroq client."""
        if self._async_client is not None:
            return self._async_client

        if not self.api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Set it in .env or pass api_key to GroqLLMClient."
            )

        try:
            from groq import AsyncGroq

            self._async_client = AsyncGroq(api_key=self.api_key)
            return self._async_client
        except Exception as exc:
            logger.error("Failed to initialize AsyncGroq client: %s", exc)
            raise

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> AsyncGenerator[Tuple[str, Optional[CompletionResult]], None]:
        """Stream chat completion yielding tokens and final telemetry upon completion.

        Yields:
            (token_text, None) during generation.
            ("", CompletionResult) as the final yielded tuple.
        """
        start_epoch = time.time()
        t0 = time.perf_counter()
        first_token_ms: Optional[float] = None
        content_parts: List[str] = []
        tokens_in = 0
        tokens_out = 0
        telemetry = RateLimitTelemetry()

        try:
            # Capture raw HTTP headers for real rate-limit metrics
            raw_response = await self.client.chat.completions.with_raw_response.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True,
            )

            telemetry = RateLimitTelemetry.from_headers(raw_response.headers)
            parsed = raw_response.parse()
            if hasattr(parsed, "__await__"):
                stream = await parsed
            else:
                stream = parsed

            async for chunk in stream:
                # Capture TTFT on first token chunk
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        if first_token_ms is None:
                            first_token_ms = (time.perf_counter() - t0) * 1000.0
                        content_parts.append(delta.content)
                        yield (delta.content, None)

                # Capture usage statistics safely
                usage = getattr(chunk, "usage", None)
                if not usage and hasattr(chunk, "x_groq") and chunk.x_groq:
                    usage = getattr(chunk.x_groq, "usage", None)

                if usage:
                    if isinstance(usage, dict):
                        p_tok = usage.get("prompt_tokens")
                        c_tok = usage.get("completion_tokens")
                    else:
                        p_tok = getattr(usage, "prompt_tokens", None)
                        c_tok = getattr(usage, "completion_tokens", None)
                    if p_tok is not None:
                        tokens_in = p_tok
                    if c_tok is not None:
                        tokens_out = c_tok

        except Exception as exc:
            if hasattr(exc, "response") and hasattr(exc.response, "headers"):
                telemetry = RateLimitTelemetry.from_headers(exc.response.headers)
            logger.error("Streaming error in GroqLLMClient: %s", exc, exc_info=True)
            raise

        t_final = time.perf_counter()
        total_ms = (t_final - t0) * 1000.0

        if first_token_ms is None:
            first_token_ms = total_ms

        full_content = "".join(content_parts)

        # Token approximation fallback if header did not supply usage
        if tokens_out == 0 and full_content:
            tokens_out = max(1, int(len(full_content.split()) * 1.3))
        if tokens_in == 0:
            tokens_in = sum(max(1, int(len(m.get("content", "").split()) * 1.3)) for m in messages)

        result = CompletionResult(
            content=full_content,
            start_time=start_epoch,
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            telemetry=telemetry,
        )

        yield ("", result)

    async def complete_stream(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> CompletionResult:
        """Fully consumes stream and returns CompletionResult with full telemetry."""
        final_result = None
        async for _, res in self.stream_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            if res is not None:
                final_result = res

        if final_result is None:
            raise RuntimeError("Stream ended without yielding final completion result.")
        return final_result