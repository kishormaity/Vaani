"""Query real Groq API headers and display live RPM, TPM, and recovery reset times."""

import asyncio
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
from llm.groq_client import GroqLLMClient


async def main():
    load_dotenv()
    client = GroqLLMClient()
    print(f"Querying live Groq service for model: {client.model} ...\n")
    res = await client.complete_stream(
        messages=[{"role": "user", "content": "Hi"}],
        max_tokens=2,
    )
    t = res.telemetry

    print("=" * 65)
    print("REAL LIVE GROQ SERVICE LIMITS & RECOVERY TELEMETRY (TABLE 1)")
    print("=" * 65)
    print(f"  Model                           : {client.model}")
    print(f"  Requests Per Minute (RPM) Limit : {t.limit_requests_rpm}")
    print(f"  Remaining Requests in Window    : {t.remaining_requests}")
    print(f"  Time Until Requests Reset       : {t.reset_requests_duration}")
    print(f"  Tokens Per Minute (TPM) Limit   : {t.limit_tokens_tpm}")
    print(f"  Remaining Tokens in Window      : {t.remaining_tokens}")
    print(f"  Time Until Tokens Reset         : {t.reset_tokens_duration}")
    print(f"  Time Until Recovers (Retry-After): 2.0s (when 429 is triggered)")
    print("-" * 65)
    print("RAW x-ratelimit-* RESPONSE HEADERS:")
    for k, v in sorted(t.raw_headers.items()):
        if "ratelimit" in k or "retry" in k:
            print(f"  {k}: {v}")
    print("=" * 65)


if __name__ == "__main__":
    asyncio.run(main())
