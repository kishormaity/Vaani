"""Test live Groq concurrency limits and measure latency degradation under concurrent load."""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from llm.groq_client import GroqLLMClient

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("concurrency_test")


async def single_worker(client: GroqLLMClient, worker_id: int):
    t0 = time.perf_counter()
    prompt = [{"role": "user", "content": "Hi"}]
    try:
        res = await client.complete_stream(messages=prompt, max_tokens=2)
        return {
            "worker_id": worker_id,
            "status": "ok",
            "ttft_ms": res.first_token_ms,
            "total_ms": res.total_ms,
            "error": None,
            "retry_after": None,
        }
    except Exception as exc:
        total_ms = (time.perf_counter() - t0) * 1000.0
        retry_after = None
        if hasattr(exc, "response") and hasattr(exc.response, "headers"):
            retry_after = exc.response.headers.get("retry-after")
        elif hasattr(exc, "retry_after"):
            retry_after = getattr(exc, "retry_after")

        return {
            "worker_id": worker_id,
            "status": "failed",
            "ttft_ms": None,
            "total_ms": total_ms,
            "error": str(exc),
            "retry_after": retry_after,
        }


async def test_batch(client: GroqLLMClient, concurrency: int):
    logger.info("=" * 60)
    logger.info(f"Testing concurrency level: {concurrency} simultaneous requests...")
    logger.info("=" * 60)

    t_start = time.perf_counter()
    tasks = [single_worker(client, i + 1) for i in range(concurrency)]
    results = await asyncio.gather(*tasks)
    batch_duration_ms = (time.perf_counter() - t_start) * 1000.0

    successful = [r for r in results if r["status"] == "ok"]
    failed = [r for r in results if r["status"] == "failed"]

    ttfts = [r["ttft_ms"] for r in successful if r["ttft_ms"] is not None]
    totals = [r["total_ms"] for r in successful]

    retry_afters = [r["retry_after"] for r in failed if r["retry_after"] is not None]

    logger.info(f"Results for Concurrency = {concurrency}:")
    logger.info(f"  Total sent: {concurrency}")
    logger.info(f"  Succeeded:  {len(successful)} ({len(successful)/concurrency*100:.1f}%)")
    logger.info(f"  Failed:     {len(failed)} ({len(failed)/concurrency*100:.1f}%)")
    logger.info(f"  Wall-clock batch time: {batch_duration_ms:.1f} ms")

    if ttfts:
        logger.info(f"  TTFT median: {np.percentile(ttfts, 50):.1f} ms | p95: {np.percentile(ttfts, 95):.1f} ms")
        logger.info(f"  Total median: {np.percentile(totals, 50):.1f} ms | p95: {np.percentile(totals, 95):.1f} ms")

    if failed:
        logger.warning("  Errors encountered:")
        for f in failed[:5]:
            logger.warning(f"    Worker {f['worker_id']}: {f['error']} (retry-after: {f['retry_after']})")

    return {
        "concurrency": concurrency,
        "succeeded": len(successful),
        "failed": len(failed),
        "ttft_p50": np.percentile(ttfts, 50) if ttfts else None,
        "ttft_p95": np.percentile(ttfts, 95) if ttfts else None,
        "retry_after": retry_afters[0] if retry_afters else None,
    }


async def main():
    client = GroqLLMClient()
    logger.info(f"Targeting model: {client.model}")

    levels = [20, 30, 40, 50]
    summary = []

    for level in levels:
        res = await test_batch(client, level)
        summary.append(res)
        logger.info("Sleeping 3.0s between concurrency bursts...")
        await asyncio.sleep(3.0)  # 3s pause to let minute token bucket recover

    logger.info("\n" + "=" * 60)
    logger.info("CONCURRENCY TEST SUMMARY (HIGHER TIERS: 20 -> 50)")
    logger.info("=" * 60)
    for s in summary:
        p50_str = f"{s['ttft_p50']:.1f} ms" if s['ttft_p50'] is not None else "N/A"
        p95_str = f"{s['ttft_p95']:.1f} ms" if s['ttft_p95'] is not None else "N/A"
        retry_str = f" | Retry-After={s['retry_after']}" if s['retry_after'] else ""
        logger.info(
            f"Concurrency {s['concurrency']:>2}: "
            f"Success={s['succeeded']:>2}/{s['concurrency']:<2} | "
            f"TTFT p50={p50_str:>9} | "
            f"TTFT p95={p95_str:>9}"
            f"{retry_str}"
        )


if __name__ == "__main__":
    asyncio.run(main())
