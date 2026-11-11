"""Token Bucket and Rolling Window Rate Limiter for The Relay LLM Gateway.

Enforces per-model constraints:
- 30 Requests Per Minute (RPM)
- 8,000 Tokens Per Minute (TPM)
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TokenBucketRateLimiter:
    """Tracks and throttles rolling 60-second request and token windows."""

    def __init__(
        self,
        rpm_limit: int = 30,
        tpm_limit: int = 8000,
        window_seconds: float = 60.0,
    ) -> None:
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.window_seconds = window_seconds

        # In-memory history of (timestamp, token_count, reservation_id)
        self._history: List[Tuple[float, int, str]] = []
        self._lock = asyncio.Lock()

    def _prune(self, now: float) -> None:
        """Prune requests older than rolling window."""
        cutoff = now - self.window_seconds
        self._history = [entry for entry in self._history if entry[0] >= cutoff]

    def acquire(self, estimated_tokens: int = 150) -> float:
        """Calculate required delay in seconds.

        Returns:
            0.0 if capacity is available immediately (reservation recorded).
            Required sleep duration (> 0.0) if constrained by RPM or TPM.
        """
        now = time.time()
        self._prune(now)

        current_requests = len(self._history)
        current_tokens = sum(tok for _, tok, _ in self._history)

        delay_rpm = 0.0
        delay_tpm = 0.0

        # Check RPM constraint
        if current_requests >= self.rpm_limit:
            # Need at least (current_requests - self.rpm_limit + 1) requests to expire
            idx_to_expire = current_requests - self.rpm_limit
            oldest_ts = self._history[idx_to_expire][0]
            delay_rpm = max(0.001, (oldest_ts + self.window_seconds) - now)

        # Check TPM constraint
        if current_tokens + estimated_tokens > self.tpm_limit:
            needed_to_expire = (current_tokens + estimated_tokens) - self.tpm_limit
            accum_tokens = 0
            t_expire = now + self.window_seconds
            for ts, tok, _ in self._history:
                accum_tokens += tok
                if accum_tokens >= needed_to_expire:
                    t_expire = ts + self.window_seconds
                    break
            delay_tpm = max(0.001, t_expire - now)

        required_delay = max(delay_rpm, delay_tpm)

        if required_delay <= 0.0:
            # Capacity available: reserve immediately
            res_id = str(uuid.uuid4())
            self._history.append((now, estimated_tokens, res_id))
            return 0.0

        # Return delay plus small jitter/margin so timestamp safely rolls past window
        return round(required_delay + 0.05, 3)

    async def wait_and_acquire(
        self,
        estimated_tokens: int = 150,
        max_wait: float = 65.0,
    ) -> float:
        """Wait asynchronously until capacity is available, then record reservation.

        Returns:
            Total delay waited in seconds.
        """
        start_time = time.perf_counter()
        while True:
            async with self._lock:
                delay = self.acquire(estimated_tokens=estimated_tokens)
                if delay <= 0.0:
                    return (time.perf_counter() - start_time)

            if (time.perf_counter() - start_time) + delay > max_wait:
                raise TimeoutError(
                    f"Rate limiter timeout: wait duration {delay:.2f}s exceeds max_wait {max_wait:.2f}s"
                )

            logger.info("Rate limit reached. Sleeping %.2fs for capacity...", delay)
            await asyncio.sleep(delay)

    def record_usage(
        self,
        actual_tokens: int,
        reservation_id: Optional[str] = None,
    ) -> None:
        """Reconcile token reservations against actual tokens consumed."""
        if not self._history:
            self._history.append((time.time(), actual_tokens, str(uuid.uuid4())))
            return

        if reservation_id is not None:
            for i, (ts, tok, r_id) in enumerate(self._history):
                if r_id == reservation_id:
                    self._history[i] = (ts, actual_tokens, r_id)
                    return

        # Default fallback: update the latest reservation in history
        last_ts, _, last_id = self._history[-1]
        self._history[-1] = (last_ts, actual_tokens, last_id)

    def get_window_stats(self) -> Dict[str, Any]:
        """Return current rolling 60s usage statistics."""
        now = time.time()
        self._prune(now)
        used_requests = len(self._history)
        used_tokens = sum(tok for _, tok, _ in self._history)
        return {
            "rpm_limit": self.rpm_limit,
            "used_requests": used_requests,
            "remaining_requests": max(0, self.rpm_limit - used_requests),
            "tpm_limit": self.tpm_limit,
            "used_tokens": used_tokens,
            "remaining_tokens": max(0, self.tpm_limit - used_tokens),
        }

    def reset(self) -> None:
        """Reset history (for tests)."""
        self._history.clear()

