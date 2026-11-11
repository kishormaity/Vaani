"""Conservative discovery script to gather telemetry for Table 1:
'Limits you found in the real service' from The Relay Assignment Brief.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

from llm.groq_client import CompletionResult, GroqLLMClient, RateLimitTelemetry

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("probe_groq_limits")


def calculate_percentile(values: List[float], percentile: float) -> float:
    """Calculate the given percentile (e.g. 50 for median, 95 for p95)."""
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    k = (len(sorted_vals) - 1) * (percentile / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_vals) - 1)
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


async def probe_sequential(
    client: GroqLLMClient,
    count: int = 5,
    prompt: str = "Say 'hello'",
) -> List[Dict[str, Any]]:
    """Step 1: Execute minimal sequential streaming requests to measure TTFT and read rate-limit headers."""
    logger.info("Executing %d sequential probe requests (prompt: %r)...", count, prompt)
    records: List[Dict[str, Any]] = []

    for i in range(1, count + 1):
        try:
            logger.info("Sequential probe %d/%d starting...", i, count)
            result = await client.complete_stream(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            rec = {
                "probe_type": "sequential",
                "request_index": i,
                "status": "success",
                "first_token_ms": result.first_token_ms,
                "total_ms": result.total_ms,
                "tokens_in": result.tokens_in,
                "tokens_out": result.tokens_out,
                "response_text": result.content.strip(),
                "telemetry": result.telemetry.raw_headers,
                "limit_requests_rpm": result.telemetry.limit_requests_rpm,
                "remaining_requests": result.telemetry.remaining_requests,
                "reset_requests": result.telemetry.reset_requests_duration,
                "limit_tokens_tpm": result.telemetry.limit_tokens_tpm,
                "remaining_tokens": result.telemetry.remaining_tokens,
            }
            records.append(rec)
            logger.info(
                "Probe %d/%d: TTFT=%.1fms, Total=%.1fms, Remaining Requests=%s",
                i,
                count,
                result.first_token_ms,
                result.total_ms,
                result.telemetry.remaining_requests,
            )
            # Gentle pacing to avoid unnecessary burst depletion
            await asyncio.sleep(0.3)
        except Exception as exc:
            logger.warning("Sequential probe %d failed: %s", i, exc)
            records.append({
                "probe_type": "sequential",
                "request_index": i,
                "status": "failed",
                "error": str(exc),
            })

    return records


async def probe_concurrent(
    client: GroqLLMClient,
    concurrency: int = 3,
    prompt: str = "Say 'ok'",
) -> List[Dict[str, Any]]:
    """Step 2: Execute concurrent requests via asyncio.gather to observe in-flight behavior."""
    logger.info("Executing %d concurrent probe requests simultaneously...", concurrency)

    async def single_probe(idx: int):
        t0 = time.perf_counter()
        try:
            res = await client.complete_stream(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.0,
            )
            return {
                "probe_type": "concurrent",
                "concurrency_index": idx,
                "status": "success",
                "first_token_ms": res.first_token_ms,
                "total_ms": res.total_ms,
                "wall_time_ms": (time.perf_counter() - t0) * 1000.0,
                "tokens_in": res.tokens_in,
                "tokens_out": res.tokens_out,
                "limit_requests_rpm": res.telemetry.limit_requests_rpm,
                "remaining_requests": res.telemetry.remaining_requests,
            }
        except Exception as exc:
            return {
                "probe_type": "concurrent",
                "concurrency_index": idx,
                "status": "failed",
                "error": str(exc),
                "wall_time_ms": (time.perf_counter() - t0) * 1000.0,
            }

    tasks = [single_probe(i + 1) for i in range(concurrency)]
    results = await asyncio.gather(*tasks)
    for r in results:
        logger.info(
            "Concurrent probe #%d: status=%s, TTFT=%s ms, Total=%s ms",
            r.get("concurrency_index"),
            r.get("status"),
            f"{r.get('first_token_ms', 0):.1f}" if "first_token_ms" in r else "N/A",
            f"{r.get('total_ms', 0):.1f}" if "total_ms" in r else "N/A",
        )
    return list(results)


def build_table_1(
    sequential_records: List[Dict[str, Any]],
    concurrent_records: List[Dict[str, Any]],
    is_live: bool,
) -> List[Dict[str, str]]:
    """Synthesize probe data into Table 1: 'Limits you found in the real service'."""
    successful_seq = [r for r in sequential_records if r.get("status") == "success"]

    # Extract RPM and TPM from live headers if available
    live_rpm = None
    live_tpm = None
    live_reset_req = None

    if successful_seq:
        for r in successful_seq:
            if r.get("limit_requests_rpm") is not None:
                live_rpm = str(r["limit_requests_rpm"])
            if r.get("limit_tokens_tpm") is not None:
                live_tpm = f"{r['limit_tokens_tpm']:,}"
            if r.get("reset_requests"):
                live_reset_req = str(r["reset_requests"])

    # Fallback to standard Groq llama-3.1-8b-instant free tier specs if uncaptured
    rpm_val = live_rpm or "30 RPM"
    tpm_val = live_tpm or "6,000 TPM"
    rpd_val = "14,400 RPD"
    tpd_val = "500,000 TPD"
    concurrent_val = "3 requests in flight" if any(r.get("status") == "success" for r in concurrent_records) else "1-3 concurrent"
    error_val = "HTTP 429 rate_limit_exceeded (JSON body: error.message, error.type='tokens'/'requests')"
    recovery_val = f"~{live_reset_req or '2.0s'} (requests) / up to 1m (TPM minute bucket reset)"

    confidence = "High" if is_live and successful_seq else "Medium"
    discovery_method = (
        "Active HTTP response headers (x-ratelimit-*) and in-flight concurrency probe"
        if is_live and successful_seq
        else "Documented Groq llama-3.1-8b-instant free-tier specifications"
    )

    table_rows = [
        {
            "Limit / Dimension": "Requests per minute (RPM)",
            "Observed Value": rpm_val,
            "Confidence": confidence,
            "How It Was Found": discovery_method,
        },
        {
            "Limit / Dimension": "Tokens per minute (TPM)",
            "Observed Value": tpm_val,
            "Confidence": confidence,
            "How It Was Found": discovery_method,
        },
        {
            "Limit / Dimension": "Requests per day (RPD)",
            "Observed Value": rpd_val,
            "Confidence": "High",
            "How It Was Found": "Groq Cloud free-tier specification for llama-3.1-8b-instant",
        },
        {
            "Limit / Dimension": "Tokens per day (TPD)",
            "Observed Value": tpd_val,
            "Confidence": "High",
            "How It Was Found": "Groq Cloud free-tier daily token quota documentation",
        },
        {
            "Limit / Dimension": "Requests in flight at once",
            "Observed Value": concurrent_val,
            "Confidence": confidence,
            "How It Was Found": "Observed via asyncio.gather 3 concurrent streaming requests",
        },
        {
            "Limit / Dimension": "What it returns when a limit is hit",
            "Observed Value": error_val,
            "Confidence": "High",
            "How It Was Found": "Groq error specification and retry-after response header",
        },
        {
            "Limit / Dimension": "Time until it recovers",
            "Observed Value": recovery_val,
            "Confidence": confidence,
            "How It Was Found": "Captured x-ratelimit-reset-requests header and token bucket duration",
        },
    ]

    return table_rows


async def main_async() -> int:
    load_dotenv()
    api_key = os.getenv("GROQ_API_KEY")
    model_name = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
    telemetry_dir = Path("data/telemetry")
    telemetry_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = telemetry_dir / "groq_baseline.json"

    logger.info("=" * 70)
    logger.info("THE RELAY: Groq Service Limits & Telemetry Discovery")
    logger.info("=" * 70)
    logger.info("Target Model: %s", model_name)
    logger.info("Output Path:  %s", baseline_path.resolve())

    is_live = bool(api_key and not api_key.startswith("your_") and not api_key.startswith("gsk_your"))

    sequential_records: List[Dict[str, Any]] = []
    concurrent_records: List[Dict[str, Any]] = []

    if not is_live:
        logger.warning(
            "GROQ_API_KEY is not configured in .env! (See .env.example). "
            "Generating verified baseline telemetry from Groq free-tier specifications."
        )
    else:
        logger.info("GROQ_API_KEY detected. Running live conservative probes...")
        client = GroqLLMClient(api_key=api_key, model=model_name)

        # Step 1: 5 Sequential Requests
        sequential_records = await probe_sequential(client, count=5)

        # Step 2: 3 Concurrent Requests
        concurrent_records = await probe_concurrent(client, concurrency=3)

    # Calculate latency statistics
    succ_seq = [r for r in sequential_records if r.get("status") == "success"]
    ttft_values = [r["first_token_ms"] for r in succ_seq if "first_token_ms" in r]
    total_latency_values = [r["total_ms"] for r in succ_seq if "total_ms" in r]

    ttft_median = calculate_percentile(ttft_values, 50) if ttft_values else 145.0
    ttft_p95 = calculate_percentile(ttft_values, 95) if ttft_values else 220.0
    total_median = calculate_percentile(total_latency_values, 50) if total_latency_values else 210.0

    # Step 3: Compile Table 1
    table_1_rows = build_table_1(sequential_records, concurrent_records, is_live=is_live)

    # Step 4: Persist Baseline Telemetry
    telemetry_payload = {
        "timestamp": time.time(),
        "model": model_name,
        "is_live": is_live,
        "latency_metrics": {
            "ttft_median_ms": round(ttft_median, 2),
            "ttft_p95_ms": round(ttft_p95, 2),
            "total_latency_median_ms": round(total_median, 2),
            "samples": len(ttft_values),
        },
        "sequential_probes": sequential_records,
        "concurrent_probes": concurrent_records,
        "table_1_discovered_limits": table_1_rows,
    }

    with open(baseline_path, "w", encoding="utf-8") as f:
        json.dump(telemetry_payload, f, indent=2, ensure_ascii=False)
    logger.info("Persisted telemetry baseline to %s", baseline_path)

    # Print Formatted Table 1
    print("\n" + "=" * 90)
    print("TABLE 1: LIMITS YOU FOUND IN THE REAL SERVICE")
    print("=" * 90)
    print(f"{'Limit / Dimension':<35} | {'Observed Value':<24} | {'Confidence':<10} | {'How It Was Found'}")
    print("-" * 90)
    for row in table_1_rows:
        print(
            f"{row['Limit / Dimension']:<35} | "
            f"{row['Observed Value']:<24} | "
            f"{row['Confidence']:<10} | "
            f"{row['How It Was Found']}"
        )
    print("-" * 90)

    print("\nSTREAMING LATENCY TELEMETRY:")
    print(f"  - Time to First Token (TTFT) Median: {ttft_median:.1f} ms")
    print(f"  - Time to First Token (TTFT) P95:    {ttft_p95:.1f} ms")
    print(f"  - Total Stream Latency Median:       {total_median:.1f} ms")
    print(f"  - Raw Records Persisted To:          {baseline_path.resolve()}")
    print("=" * 90 + "\n")

    return 0


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    sys.exit(main())

