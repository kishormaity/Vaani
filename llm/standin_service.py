"""Stand-in Model Service calibrated to real Groq service measurements.

Provides an asynchronous streaming LLM client with realistic TTFT jitter,
inter-token delay, in-memory rate/token limiting (RPM, TPM, RPD, TPD),
and concurrency-based degradation.
"""

from __future__ import annotations

import asyncio
import logging
import math
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple

from llm.base import CompletionResult, LLMClientProtocol, RateLimitError, RateLimitTelemetry

logger = logging.getLogger(__name__)

try:
    import tiktoken

    _ENCODING = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODING = None


def estimate_tokens(text: str) -> int:
    """Accurately estimate token count using tiktoken (cl100k_base) or whitespace fallback."""
    if not text:
        return 0
    if _ENCODING is not None:
        try:
            return len(_ENCODING.encode(text, disallowed_special=()))
        except Exception:
            pass
    return max(1, len(text.split()))


@dataclass
class StandinConfig:
    """Configuration parameters for StandinLLMService calibrated to Table 1 measurements."""

    # Latency Calibration (Empirical baseline: median ~613.0 ms, p95 ~2565.0 ms)
    base_ttft_ms: float = 613.0
    p95_ttft_ms: float = 2565.0
    ms_per_token: float = 3.5  # Streaming inter-token delay (~2.5 - 5.0 ms/token)

    # Rate and Token Limits (Empirical baseline: RPM 1000, TPM 8000)
    rpm_limit: int = 1000  # Requests per minute limit
    tpm_limit: int = 8000  # Tokens per minute limit
    rpd_limit: int = 14400  # Requests per day limit
    tpd_limit: int = 500000  # Tokens per day limit

    # Concurrency and Overload
    max_concurrency: int = 3  # In-flight request concurrency threshold
    overload_multiplier: float = 1.5  # Latency scale factor per request above max_concurrency

    # Reliability
    failure_rate: float = 0.02  # Transient 500/503 error probability (0.0 for tests)

    # Simulation Controls
    simulate_latency: bool = True  # If False, computes latency mathematically without asyncio.sleep
    fixed_response: Optional[str] = None  # If set, always returns this exact response string
    seed: Optional[int] = None  # For deterministic reproducibility in tests


class StandinLLMService:
    """Simulated LLM service exposing identical streaming interface as GroqLLMClient."""

    source: str = "stand-in"

    def __init__(self, config: Optional[StandinConfig] = None) -> None:
        self.config = config or StandinConfig()
        self.source = "stand-in"
        self._rng = random.Random(self.config.seed)

        # Sliding window for minute rate limits: list of (epoch_timestamp, token_count)
        self._minute_window: List[Tuple[float, int]] = []

        # Daily cumulative counters
        self._daily_requests: int = 0
        self._daily_tokens: int = 0

        # In-flight concurrency tracking
        self._active_requests: int = 0
        self._lock = asyncio.Lock()

        # Precompute lognormal parameters for TTFT sampling
        self._init_ttft_distribution()

    def _init_ttft_distribution(self) -> None:
        """Calculate lognormal mu and sigma matching base_ttft_ms (median) and p95_ttft_ms."""
        base = max(1.0, self.config.base_ttft_ms)
        p95 = max(base, self.config.p95_ttft_ms)
        self._mu = math.log(base)
        # In a standard normal distribution, Z_0.95 ≈ 1.6448536
        if p95 > base:
            self._sigma = (math.log(p95) - self._mu) / 1.6448536
        else:
            self._sigma = 0.0

    def sample_ttft(self) -> float:
        """Sample a realistic TTFT in milliseconds from the calibrated lognormal distribution."""
        if self._sigma <= 0.0:
            return self.config.base_ttft_ms
        z = self._rng.gauss(0.0, 1.0)
        sample = math.exp(self._mu + self._sigma * z)
        return max(5.0, sample)

    def reset_limits(self) -> None:
        """Reset sliding window and daily counters (useful for unit tests)."""
        self._minute_window.clear()
        self._daily_requests = 0
        self._daily_tokens = 0
        self._active_requests = 0

    @property
    def active_requests(self) -> int:
        return self._active_requests

    @property
    def daily_requests(self) -> int:
        return self._daily_requests

    @property
    def daily_tokens(self) -> int:
        return self._daily_tokens

    def _prune_minute_window(self, now: float) -> None:
        """Remove requests older than 60 seconds from the minute sliding window."""
        cutoff = now - 60.0
        self._minute_window = [entry for entry in self._minute_window if entry[0] >= cutoff]

    def _calculate_concurrency_multiplier(self) -> float:
        """Compute latency degradation multiplier based on current in-flight concurrency."""
        if self._active_requests > self.config.max_concurrency:
            excess = self._active_requests - self.config.max_concurrency
            return self.config.overload_multiplier ** excess
        return 1.0

    def _generate_synthetic_response(self, messages: List[Dict[str, str]]) -> str:
        """Synthesize a realistic, citation-grounded response based on prompt context."""
        if self.config.fixed_response is not None:
            return self.config.fixed_response

        # Concatenate prompt content
        prompt_text = " ".join(m.get("content", "") for m in messages)
        lower_prompt = prompt_text.lower()

        # Check user question content specifically for unanswerable question triggers
        user_content = " ".join(m.get("content", "") for m in messages if m.get("role") == "user").lower()
        if not user_content:
            user_content = lower_prompt

        # Check if this is a DIRECT_LLM conversation call (bypassing RAG context)
        if any("helpful ai assistant" in m.get("content", "").lower() for m in messages):
            if "who are you" in user_content or "introduce yourself" in user_content:
                return "I am The Relay, an intelligent research and dialogue assistant designed to answer questions accurately and concisely."
            elif any(g in user_content for g in ("hello", "hi", "hey", "greetings")):
                return "Hello! I am The Relay, your AI assistant. How can I help you today?"
            elif "prime" in user_content:
                return "A prime number is a natural number greater than 1 that cannot be formed by multiplying two smaller natural numbers."
            else:
                return "I am ready to assist you with general knowledge, programming tasks, or research paper queries."

        unanswerable_triggers = ["quantum", "whisper", "bert", "wmt 2024", "negative", "not in context"]
        if any(trigger in user_content for trigger in unanswerable_triggers):
            return "The provided passages do not contain sufficient information to answer this question."

        # Extract citations strictly from retrieved context passages (ignoring system prompt examples)
        context_citations = re.findall(r"(?:Required Citation Tag|Citation):\s*(\[arXiv:[^\]]+\])", prompt_text)
        if not context_citations:
            all_cits = re.findall(r"\[arXiv:[^\]]+\]", prompt_text)
            # Filter out the example citation tag from system prompt
            context_citations = [c for c in all_cits if "p.5" not in c or "1706.03762" not in c or "Context Passages" not in prompt_text]
            if not context_citations:
                context_citations = all_cits

        citations_unique = sorted(list(set(context_citations)))

        if not citations_unique:
            return "The provided passages do not contain sufficient information to answer this question."

        # Grounded synthetic response incorporating detected citations
        citation_str = ", ".join(citations_unique[:2])

        if "scaled dot-product" in lower_prompt or "formula" in lower_prompt and "attention" in lower_prompt:
            return (
                f"Scaled Dot-Product Attention is computed as Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V {citation_str}. "
                f"The scaling factor 1/sqrt(d_k) is applied because for large values of d_k, the dot products grow large in magnitude, "
                f"pushing the softmax function into regions with extremely small gradients."
            )
        elif "hyperparameter" in lower_prompt or "base transformer" in lower_prompt:
            return (
                f"In the base Transformer model {citation_str}, both the encoder and decoder have N = 6 layers, "
                f"with model dimension d_model = 512, inner feed-forward dimension d_ff = 2048, and h = 8 parallel attention heads "
                f"with d_k = d_v = 64."
            )
        elif "conversational latency" in lower_prompt or ("latency" in lower_prompt and "moshi" in lower_prompt):
            return (
                f"Moshi achieves a theoretical conversational latency of 160 ms (comprising an 80 ms audio frame size "
                f"plus an 80 ms acoustic lookahead) and an overall practical latency of approximately 200 ms under deployed conditions {citation_str}."
            )
        elif "mimi" in lower_prompt or ("codec" in lower_prompt and "audio" in lower_prompt):
            return (
                f"Moshi employs the Mimi neural audio codec {citation_str}, which tokenizes 24kHz audio into discrete tokens "
                f"operating at a frame rate of 12.5 Hz and an audio bandwidth/bit rate of 1.1 kbps."
            )
        elif "segment boundaries" in lower_prompt or "token boundaries" in lower_prompt or "dynamic chunking" in lower_prompt:
            return (
                f"Dynamic Chunking identifies token segment boundaries dynamically {citation_str} using transition entropy "
                f"and semantic coherence rather than relying on traditional fixed-size token window splitting."
            )
        elif "multi-stream" in lower_prompt:
            return (
                f"Moshi handles simultaneous text and audio streaming through multi-stream modeling {citation_str}, "
                f"interleaving text tokens and hierarchical audio tokens across parallel codebooks for full-duplex conversational interaction."
            )
        elif "compare" in lower_prompt or "difference" in lower_prompt:
            return (
                f"Comparing the architectures {citation_str}, the Transformer uses non-causal multi-head attention for offline sequence-to-sequence modeling, "
                f"whereas Moshi introduces streaming causal multi-stream modeling for full-duplex audio, and Dynamic Chunking optimizes boundary segmentation."
            )
        elif "attention" in lower_prompt or "transformer" in lower_prompt:
            return (
                f"Based on the provided research context {citation_str}, the Transformer architecture relies on multi-head "
                f"self-attention mechanisms to compute representations of input sequences without recurrent or convolutional layers."
            )
        elif "moshi" in lower_prompt or "speech" in lower_prompt:
            return (
                f"According to the provided document {citation_str}, the full-duplex spoken dialogue "
                f"framework employs joint text-to-speech and speech-to-text token modeling to achieve "
                f"low conversational latency."
            )
        else:
            return (
                f"Based on the cited evidence {citation_str}, the documented method optimizes "
                f"both performance and computational efficiency across the evaluated benchmarks."
            )

    async def stream_chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 150,
        temperature: float = 0.0,
    ) -> AsyncGenerator[Tuple[str, Optional[CompletionResult]], None]:
        """Stream simulated chat completion with calibrated delays, telemetry, and rate limits.

        Yields:
            (token_text, None) during streaming.
            ("", CompletionResult) as the final yielded tuple.
        """
        now = time.time()
        start_epoch = now
        t0 = time.perf_counter()

        # Step 1: Input token estimation
        tokens_in = sum(estimate_tokens(m.get("content", "")) for m in messages)

        # Step 2: Rate and quota limit validation under lock
        async with self._lock:
            # Check Daily Limits (RPD / TPD)
            if self._daily_requests >= self.config.rpd_limit:
                raise RateLimitError(
                    f"Daily request quota exceeded: RPD limit of {self.config.rpd_limit} reached",
                    retry_after=86400.0,
                    status_code=429,
                    limit_type="RPD",
                )
            if self._daily_tokens >= self.config.tpd_limit:
                raise RateLimitError(
                    f"Daily token quota exceeded: TPD limit of {self.config.tpd_limit} reached",
                    retry_after=86400.0,
                    status_code=429,
                    limit_type="TPD",
                )

            # Check Minute Sliding Window Limits (RPM / TPM)
            self._prune_minute_window(now)
            current_rpm = len(self._minute_window)
            current_tpm = sum(tok for _, tok in self._minute_window)

            if current_rpm >= self.config.rpm_limit:
                oldest_time = self._minute_window[0][0]
                retry_after = max(0.1, round(60.0 - (now - oldest_time), 2))
                raise RateLimitError(
                    f"Rate limit exceeded: RPM limit of {self.config.rpm_limit} reached",
                    retry_after=retry_after,
                    status_code=429,
                    limit_type="RPM",
                )

            if current_tpm + tokens_in > self.config.tpm_limit:
                oldest_time = self._minute_window[0][0] if self._minute_window else now
                retry_after = max(0.1, round(60.0 - (now - oldest_time), 2))
                raise RateLimitError(
                    f"Rate limit exceeded: TPM limit of {self.config.tpm_limit} reached",
                    retry_after=retry_after,
                    status_code=429,
                    limit_type="TPM",
                )

            # Check transient failure simulation
            if self.config.failure_rate > 0.0 and self._rng.random() < self.config.failure_rate:
                raise RuntimeError("503 Service Unavailable: Simulated transient upstream error")

            # Register in-flight request
            self._active_requests += 1

        try:
            # Step 3: Concurrency degradation calculation
            concurrency_mult = self._calculate_concurrency_multiplier()
            sampled_ttft = self.sample_ttft()
            effective_ttft_ms = sampled_ttft * concurrency_mult
            effective_ms_per_token = self.config.ms_per_token * concurrency_mult

            # Generate synthetic content
            full_text = self._generate_synthetic_response(messages)
            # Tokenize into words / whitespace-preserving chunks
            raw_tokens = re.findall(r"\S+|\s+", full_text)
            if not raw_tokens:
                raw_tokens = [full_text]

            tokens_out = estimate_tokens(full_text)
            if max_tokens and tokens_out > max_tokens:
                # Truncate tokens if max_tokens exceeded
                raw_tokens = raw_tokens[:max_tokens]
                full_text = "".join(raw_tokens)
                tokens_out = estimate_tokens(full_text)

            # Step 4: Stream TTFT Delay
            if self.config.simulate_latency:
                await asyncio.sleep(effective_ttft_ms / 1000.0)
                first_token_ms = (time.perf_counter() - t0) * 1000.0
            else:
                first_token_ms = effective_ttft_ms

            # Step 5: Yield streaming tokens
            for i, token in enumerate(raw_tokens):
                if i > 0 and self.config.simulate_latency and effective_ms_per_token > 0:
                    await asyncio.sleep(effective_ms_per_token / 1000.0)
                yield (token, None)

            # Step 6: Final latency calculation
            if self.config.simulate_latency:
                total_ms = (time.perf_counter() - t0) * 1000.0
            else:
                inter_token_total = max(0, len(raw_tokens) - 1) * effective_ms_per_token
                total_ms = effective_ttft_ms + inter_token_total

        finally:
            # Step 7: Record completed request & update state under lock
            async with self._lock:
                self._active_requests = max(0, self._active_requests - 1)
                total_request_tokens = tokens_in + tokens_out
                completion_time = time.time()
                self._minute_window.append((completion_time, total_request_tokens))
                self._daily_requests += 1
                self._daily_tokens += total_request_tokens

                # Calculate telemetry snapshots
                self._prune_minute_window(completion_time)
                rem_req = max(0, self.config.rpm_limit - len(self._minute_window))
                rem_tok = max(
                    0, self.config.tpm_limit - sum(tok for _, tok in self._minute_window)
                )
                oldest_t = self._minute_window[0][0] if self._minute_window else completion_time
                reset_secs = int(max(0, 60.0 - (completion_time - oldest_t)))

        # Step 8: Build telemetry object
        telemetry = RateLimitTelemetry(
            limit_requests_rpm=self.config.rpm_limit,
            remaining_requests=rem_req,
            reset_requests_duration=f"{reset_secs}s",
            limit_tokens_tpm=self.config.tpm_limit,
            remaining_tokens=rem_tok,
            reset_tokens_duration=f"{reset_secs}s",
            retry_after=None,
            raw_headers={
                "x-ratelimit-limit-requests": str(self.config.rpm_limit),
                "x-ratelimit-remaining-requests": str(rem_req),
                "x-ratelimit-reset-requests": f"{reset_secs}s",
                "x-ratelimit-limit-tokens": str(self.config.tpm_limit),
                "x-ratelimit-remaining-tokens": str(rem_tok),
                "x-ratelimit-reset-tokens": f"{reset_secs}s",
            },
        )

        result = CompletionResult(
            content=full_text,
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
        """Convenience method that fully consumes the stream and returns CompletionResult."""
        final_result = None
        async for _, res in self.stream_chat(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        ):
            if res is not None:
                final_result = res

        if final_result is None:
            raise RuntimeError("StandinLLMService stream completed without yielding final result.")
        return final_result

