"""Interactive Real-Time Terminal CLI for The Relay RAG Pipeline & LLM Gateway.

Allows users to test, inspect, and evaluate queries in real time with detailed
breakdowns of intent classification, retrieval provenance, citation verification,
synthesized answers, and performance telemetry.

Usage:
  Interactive loop:
    python scripts/cli.py
    python scripts/cli.py --setting C --live
    python scripts/cli.py --standin

  One-shot query:
    python scripts/cli.py "Explain scaled dot-product attention"
    python scripts/cli.py --setting B "Who are you?"
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load .env if present
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from gateway.llm_gateway import LLMGateway
from llm.groq_client import GroqLLMClient
from llm.standin_service import StandinConfig, StandinLLMService
from pipeline.classifier import QueryIntent
from pipeline.orchestrator import PipelineResult, RAGOrchestrator
from retrieval.retriever import RetrievedChunk, Retriever

# =====================================================================
# ANSI Color Palette for Enhanced Terminal Output
# =====================================================================
class Style:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"

    # Foreground colors
    BLACK = "\033[30m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"

    # Bright foreground
    BRIGHT_RED = "\033[91m"
    BRIGHT_GREEN = "\033[92m"
    BRIGHT_YELLOW = "\033[93m"
    BRIGHT_BLUE = "\033[94m"
    BRIGHT_MAGENTA = "\033[95m"
    BRIGHT_CYAN = "\033[96m"
    BRIGHT_WHITE = "\033[97m"


def enable_windows_ansi() -> None:
    """Enable virtual terminal processing on Windows if supported."""
    if os.name == "nt":
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetStdHandle(-11)
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(handle, ctypes.byref(mode))
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
        except Exception:
            pass


# =====================================================================
# Pretty Printing & Inspection Formatters
# =====================================================================
def print_banner(mode: str, setting: str, model_rag: str, model_chat: str) -> None:
    print(f"\n{Style.BRIGHT_CYAN}" + "=" * 80)
    print(f" {Style.BOLD}THE RELAY (VAANI) — REAL-TIME RAG & GATEWAY TERMINAL CLI{Style.RESET}{Style.BRIGHT_CYAN}")
    print("=" * 80 + Style.RESET)
    print(f" {Style.BOLD}Engine Mode:{Style.RESET}    {Style.BRIGHT_GREEN}{mode}{Style.RESET}")
    print(f" {Style.BOLD}Retrieval:{Style.RESET}      {Style.BRIGHT_YELLOW}{setting}{Style.RESET}")
    print(f" {Style.BOLD}RAG Model:{Style.RESET}      {Style.WHITE}{model_rag}{Style.RESET}")
    print(f" {Style.BOLD}Chat Model:{Style.RESET}     {Style.WHITE}{model_chat}{Style.RESET}")
    print(f" {Style.BOLD}Target Corpus:{Style.RESET}  Attention (1706.03762), Moshi (2410.00037), Dynamic Chunking (2507.07955)")
    print(f"{Style.BRIGHT_CYAN}" + "-" * 80 + Style.RESET)


def format_preview(text: str, max_chars: int = 150) -> str:
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "..."


def display_pipeline_result(result: PipelineResult, verbose: bool = True) -> None:
    """Format and print all pipeline stages cleanly to stdout."""
    clf = result.classification
    intent_val = clf.intent.value if clf and hasattr(clf.intent, "value") else (str(clf.intent) if clf else "UNKNOWN")

    # 1. Routing & Intent
    print(f"\n{Style.BOLD}{Style.BRIGHT_BLUE}[1] ROUTING & INTENT CLASSIFICATION{Style.RESET}")
    print(f"{Style.DIM}" + "-" * 78 + f"{Style.RESET}")
    
    intent_color = (
        Style.BRIGHT_GREEN if intent_val == "CORPUS_RAG"
        else (Style.BRIGHT_CYAN if intent_val == "DIRECT_LLM" else Style.BRIGHT_YELLOW)
    )
    print(f"  {Style.BOLD}• Detected Route:{Style.RESET}      {intent_color}{Style.BOLD}{intent_val}{Style.RESET}")
    if clf:
        print(f"  {Style.BOLD}• Question Type:{Style.RESET}       {clf.question_type}")
        print(f"  {Style.BOLD}• In Scope:{Style.RESET}            {clf.is_in_scope}")
        if clf.target_papers:
            print(f"  {Style.BOLD}• Target Paper(s):{Style.RESET}     {', '.join(clf.target_papers)}")
        if clf.reasoning:
            print(f"  {Style.BOLD}• Classifier Reason:{Style.RESET}   {Style.DIM}{clf.reasoning}{Style.RESET}")
        if clf.external_url:
            print(f"  {Style.BOLD}• Target URL:{Style.RESET}          {Style.UNDERLINE}{clf.external_url}{Style.RESET}")

    # 2. Retrieved Chunks (if CORPUS_RAG)
    if intent_val == "CORPUS_RAG" and result.chunks:
        print(f"\n{Style.BOLD}{Style.BRIGHT_BLUE}[2] RETRIEVED CORPUS CHUNKS ({len(result.chunks)} passages){Style.RESET}")
        print(f"{Style.DIM}" + "-" * 78 + f"{Style.RESET}")
        for idx, chunk in enumerate(result.chunks, 1):
            pages_str = ", ".join(str(p) for p in chunk.page_numbers)
            print(
                f"  {Style.BRIGHT_MAGENTA}#{idx}{Style.RESET} "
                f"{Style.BOLD}{chunk.citation}{Style.RESET} | "
                f"Score: {Style.BRIGHT_YELLOW}{chunk.score:.4f}{Style.RESET} | "
                f"Pages: {pages_str}"
            )
            print(f"      {Style.DIM}Paper:{Style.RESET} {chunk.paper_title}")
            print(f"      {Style.DIM}Text:{Style.RESET}  {format_preview(chunk.text, 140)}")
            print()

    # 3. Verification Details (if applicable)
    v_state = result.verification_state
    if v_state and intent_val == "CORPUS_RAG":
        print(f"{Style.BOLD}{Style.BRIGHT_BLUE}[3] CITATION VERIFICATION & REVISION{Style.RESET}")
        print(f"{Style.DIM}" + "-" * 78 + f"{Style.RESET}")
        if v_state.is_refusal:
            status_str = f"{Style.BRIGHT_YELLOW}SAFE REFUSAL TRIGGERED{Style.RESET}"
        elif v_state.is_valid:
            status_str = f"{Style.BRIGHT_GREEN}GROUNDING VERIFIED & ACCEPTED{Style.RESET}"
        else:
            status_str = f"{Style.BRIGHT_RED}INVALID CITATION DETECTED{Style.RESET}"
        
        print(f"  {Style.BOLD}• Status:{Style.RESET}              {status_str}")
        print(f"  {Style.BOLD}• Correction Retries:{Style.RESET}  {result.loop_count}")
        if v_state.valid_citations:
            cits = ", ".join(v_state.valid_citations)
            print(f"  {Style.BOLD}• Valid Citations:{Style.RESET}     {Style.GREEN}{cits}{Style.RESET}")
        if v_state.invalid_citations:
            inv = ", ".join(v_state.invalid_citations)
            print(f"  {Style.BOLD}• Invalid Citations:{Style.RESET}   {Style.BRIGHT_RED}{inv}{Style.RESET}")
        if v_state.error_message:
            print(f"  {Style.BOLD}• Verifier Feedback:{Style.RESET}   {Style.DIM}{v_state.error_message}{Style.RESET}")

    # 4. Final Answer
    print(f"\n{Style.BOLD}{Style.BRIGHT_BLUE}[4] SYNTHESIZED RESPONSE{Style.RESET}")
    print(f"{Style.DIM}" + "-" * 78 + f"{Style.RESET}")
    print(f"{Style.BRIGHT_WHITE}{result.answer.strip()}{Style.RESET}")

    # 5. Verified Citations List
    if v_state and v_state.valid_citations:
        print(f"\n{Style.BOLD}{Style.BRIGHT_BLUE}[5] VERIFIED ACADEMIC CITATIONS{Style.RESET}")
        print(f"{Style.DIM}" + "-" * 78 + f"{Style.RESET}")
        for c in set(v_state.valid_citations):
            # Attempt to find corresponding chunk title
            matching_title = next(
                (ch.paper_title for ch in result.chunks if ch.citation == c or ch.paper_id in c),
                "Indexed Research Paper"
            )
            print(f"  {Style.GREEN}✓{Style.RESET} {Style.BOLD}{c}{Style.RESET} — {Style.DIM}{matching_title}{Style.RESET}")

    # 6. Telemetry & Execution Breakdown
    telem = result.telemetry or {}
    print(f"\n{Style.BOLD}{Style.BRIGHT_BLUE}[6] EXECUTION TELEMETRY{Style.RESET}")
    print(f"{Style.DIM}" + "-" * 78 + f"{Style.RESET}")
    steps_str = " -> ".join(result.steps_executed) if result.steps_executed else "N/A"
    print(f"  {Style.BOLD}• Pipeline Steps:{Style.RESET}      {Style.DIM}{steps_str}{Style.RESET}")
    print(f"  {Style.BOLD}• Model Deployed:{Style.RESET}      {telem.get('model', 'N/A')} ({telem.get('source', 'unknown')})")
    
    ret_ms = telem.get("retrieval_ms", 0.0)
    ttft_ms = telem.get("first_token_ms", 0.0)
    tot_ms = telem.get("total_ms", 0.0)
    e2e_ms = telem.get("end_to_end_ms", 0.0)
    web_ms = telem.get("web_fetch_ms", None)

    perf_parts = []
    if ret_ms > 0:
        perf_parts.append(f"Retrieval: {ret_ms:.1f}ms")
    if web_ms:
        perf_parts.append(f"Web Fetch: {web_ms:.1f}ms")
    if ttft_ms > 0:
        perf_parts.append(f"TTFT: {ttft_ms:.1f}ms")
    if tot_ms > 0:
        perf_parts.append(f"LLM Gen: {tot_ms:.1f}ms")
    perf_parts.append(f"End-to-End: {e2e_ms:.1f}ms")

    print(f"  {Style.BOLD}• Latency Profile:{Style.RESET}     {' | '.join(perf_parts)}")
    
    tokens_in = telem.get("tokens_in", 0)
    tokens_out = telem.get("tokens_out", 0)
    if tokens_in or tokens_out:
        print(f"  {Style.BOLD}• Token Usage:{Style.RESET}         Prompt: {tokens_in} tokens | Completion: {tokens_out} tokens")
    print(f"{Style.BRIGHT_CYAN}" + "=" * 80 + f"{Style.RESET}\n")


# =====================================================================
# Core Setup and Execution
# =====================================================================
def build_orchestrator(
    setting_flag: str,
    use_standin: bool,
    model_override: Optional[str] = None,
    chat_model_override: Optional[str] = None,
) -> Tuple[RAGOrchestrator, str, str, str, str]:
    """Construct RAGOrchestrator with the specified settings and model backend."""
    # Normalize setting string
    clean_setting = setting_flag.strip().upper()
    if clean_setting in ("A", "SETTING A"):
        setting = "Setting A"
    elif clean_setting in ("B", "SETTING B"):
        setting = "Setting B"
    elif clean_setting in ("C", "SETTING C"):
        setting = "Setting C"
    else:
        setting = "Setting C"

    rag_model = model_override or os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    chat_model = chat_model_override or os.getenv("GROQ_CHAT_MODEL", "qwen/qwen3.8-27b")

    groq_api_key = os.getenv("GROQ_API_KEY", "").strip()

    if use_standin or not groq_api_key:
        mode_label = "STAND-IN (Offline Deterministic Simulation)"
        standin = StandinLLMService(
            config=StandinConfig(
                simulate_latency=False,
                failure_rate=0.0,
                rpm_limit=10000,
                tpm_limit=2000000,
            )
        )
        gateway = LLMGateway(client=standin)
        retriever = Retriever(setting=setting)
        orchestrator = RAGOrchestrator(
            retriever=retriever,
            llm_client=standin,
            chat_llm_client=standin,
            gateway=gateway,
            chat_gateway=gateway,
            setting=setting,
            chat_model=chat_model,
        )
    else:
        mode_label = "LIVE (Groq API High-Speed Cloud Inference)"
        groq_rag_client = GroqLLMClient(model=rag_model)
        groq_chat_client = GroqLLMClient(model=chat_model)

        gateway = LLMGateway(client=groq_rag_client)
        chat_gateway = LLMGateway(client=groq_chat_client)

        retriever = Retriever(setting=setting)
        orchestrator = RAGOrchestrator(
            retriever=retriever,
            llm_client=groq_rag_client,
            chat_llm_client=groq_chat_client,
            gateway=gateway,
            chat_gateway=chat_gateway,
            setting=setting,
            chat_model=chat_model,
        )

    return orchestrator, mode_label, setting, rag_model, chat_model


async def run_single_query(orchestrator: RAGOrchestrator, query: str) -> None:
    """Execute a single query through the pipeline and display formatted results."""
    print(f"\n{Style.DIM}Running pipeline for query: {Style.RESET}{Style.BOLD}'{query}'{Style.RESET}...")
    try:
        res = await orchestrator.run_pipeline(query)
        display_pipeline_result(res)
    except Exception as exc:
        print(f"\n{Style.BRIGHT_RED}[ERROR] Pipeline failed during query processing: {exc}{Style.RESET}")


async def interactive_loop(orchestrator: RAGOrchestrator) -> None:
    """Run interactive REPL loop."""
    print(f"{Style.BOLD}Enter your question at the prompt.{Style.RESET}")
    print(f"{Style.DIM}Type 'exit', 'quit', or 'q' to leave the session.{Style.RESET}\n")

    while True:
        try:
            query = input(f"{Style.BRIGHT_GREEN}{Style.BOLD}Vaani > {Style.RESET}").strip()
        except (KeyboardInterrupt, EOFError):
            print(f"\n\n{Style.DIM}Exiting session. Goodbye!{Style.RESET}")
            break

        if not query:
            continue

        if query.lower() in ("exit", "quit", "q"):
            print(f"\n{Style.DIM}Exiting session. Goodbye!{Style.RESET}")
            break

        await run_single_query(orchestrator, query)


# =====================================================================
# CLI Entry Point
# =====================================================================
def main() -> None:
    enable_windows_ansi()

    parser = argparse.ArgumentParser(
        description="The Relay (Vaani) RAG & LLM Gateway Interactive CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "query",
        nargs="?",
        default=None,
        help="Optional single query to execute. If omitted, starts interactive REPL mode.",
    )
    parser.add_argument(
        "--setting",
        "-s",
        default="C",
        choices=["A", "B", "C", "Setting A", "Setting B", "Setting C"],
        help="Retrieval configuration setting: A (k=3), B (k=5), or C (k_cand=8 -> k=3 CrossEncoder reranked). Default: C",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Force using live Groq API endpoints.",
    )
    parser.add_argument(
        "--standin",
        action="store_true",
        help="Use offline calibrated StandinLLMService (zero quota, deterministic simulation).",
    )
    parser.add_argument(
        "--model",
        "-m",
        default=None,
        help="Override Groq model for CORPUS_RAG (default: openai/gpt-oss-20b).",
    )
    parser.add_argument(
        "--chat-model",
        default=None,
        help="Override Groq model for DIRECT_LLM (default: qwen/qwen3.8-27b).",
    )

    args = parser.parse_args()

    # Determine whether to use stand-in
    use_standin = args.standin or (not args.live and not os.getenv("GROQ_API_KEY"))

    orchestrator, mode_label, setting_name, rag_model, chat_model = build_orchestrator(
        setting_flag=args.setting,
        use_standin=use_standin,
        model_override=args.model,
        chat_model_override=args.chat_model,
    )

    print_banner(mode_label, setting_name, rag_model, chat_model)

    if args.query:
        # One-shot mode
        asyncio.run(run_single_query(orchestrator, args.query))
    else:
        # Interactive loop
        asyncio.run(interactive_loop(orchestrator))


if __name__ == "__main__":
    main()

