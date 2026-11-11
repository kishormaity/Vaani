"""4-Step RAG Pipeline Orchestrator with Multi-Route Query Dispatch and Verification Loop.

Implements query routing across:
- CORPUS_RAG: Step 1 (Analyze) -> Step 2 (Retriever) -> Step 3 (Draft) -> Step 4 (Verify/Loop)
- DIRECT_LLM: Step 1 (Analyze) -> LLM Gateway direct stream (bypassing retriever & verifier)
- EXTERNAL_WEB: Step 1 (Analyze) -> fetch_web_page tool -> LLM Gateway synthesis
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from gateway.llm_gateway import LLMGateway
from llm.base import CompletionResult, LLMClientProtocol
from pipeline.answer_generator import AnswerGenerator, GenerationResult
from pipeline.citation_verifier import CitationVerifier, VerificationResult
from pipeline.classifier import ClassificationResult, QueryClassifier, QueryIntent
from retrieval.retriever import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)

SAFE_OUT_OF_SCOPE_REFUSAL = (
    "This query is out of scope. I can only answer questions based on the three "
    "verified research papers: 'Attention Is All You Need' (arXiv:1706.03762), "
    "'Moshi: a speech-text foundation model for real-time dialogue' (arXiv:2410.00037), "
    "and 'Dynamic Chunking for Retrieval-Augmented Generation' (arXiv:2507.07955)."
)

SAFE_VALIDATION_FAILURE_REFUSAL = (
    "The provided research papers do not contain sufficient evidence to answer this question."
)

DIRECT_SYSTEM_PROMPT = (
    "You are The Relay, an intelligent and helpful AI assistant. "
    "Provide clear, accurate, and concise answers to general queries, "
    "programming tasks, and common knowledge questions."
)

WEB_SYSTEM_PROMPT = (
    "You are The Relay, an intelligent web-assisted research assistant. "
    "Synthesize a clear, accurate, and concise answer to the user's question "
    "based strictly on the retrieved web content provided below."
)


@dataclass
class PipelineResult:
    """Structured output from the RAG / Gateway pipeline."""

    answer: str
    chunks: List[RetrievedChunk]
    verification_state: VerificationResult
    loop_count: int
    telemetry: Dict[str, Any] = field(default_factory=dict)
    classification: Optional[ClassificationResult] = None
    steps_executed: List[str] = field(default_factory=list)

    # Convenience properties for backwards compatibility
    @property
    def final_answer(self) -> str:
        return self.answer

    @property
    def retrieved_chunks(self) -> List[RetrievedChunk]:
        return self.chunks

    @property
    def validation_result(self) -> VerificationResult:
        return self.verification_state

    @property
    def retry_count(self) -> int:
        return self.loop_count

    @property
    def is_refusal(self) -> bool:
        if self.answer in (SAFE_OUT_OF_SCOPE_REFUSAL, SAFE_VALIDATION_FAILURE_REFUSAL):
            return True
        return self.verification_state.is_refusal if self.verification_state else False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer": self.answer,
            "chunks": [c.to_dict() for c in self.chunks],
            "verification_state": self.verification_state.to_dict() if self.verification_state else None,
            "loop_count": self.loop_count,
            "telemetry": self.telemetry,
            "classification": self.classification.to_dict() if self.classification else None,
            "steps_executed": self.steps_executed,
        }


# Backwards compatibility aliases
OrchestratorResult = PipelineResult


class RAGOrchestrator:
    """Coordinates multi-route query dispatching and RAG verification loops."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm_client: Optional[LLMClientProtocol] = None,
        chat_llm_client: Optional[LLMClientProtocol] = None,
        classifier: Optional[QueryClassifier] = None,
        citation_verifier: Optional[CitationVerifier] = None,
        answer_generator: Optional[AnswerGenerator] = None,
        setting: str = "Setting A",
        gateway: Optional[LLMGateway] = None,
        chat_gateway: Optional[LLMGateway] = None,
        chat_model: Optional[str] = None,
    ) -> None:
        from llm import GroqLLMClient

        self.setting = setting
        self.retriever = retriever or Retriever(setting=setting)

        if llm_client is None:
            self.llm_client: LLMClientProtocol = GroqLLMClient()
        else:
            self.llm_client = llm_client

        if gateway is not None:
            self.gateway = gateway
        elif isinstance(self.llm_client, LLMGateway):
            self.gateway = self.llm_client
        else:
            self.gateway = LLMGateway(client=self.llm_client)

        # General conversation / DIRECT_LLM model and gateway
        self.chat_model = chat_model or os.getenv("GROQ_CHAT_MODEL", "qwen/qwen3.8-27b")

        if chat_gateway is not None:
            self.chat_gateway = chat_gateway
        elif chat_llm_client is not None:
            if isinstance(chat_llm_client, LLMGateway):
                self.chat_gateway = chat_llm_client
            else:
                self.chat_gateway = LLMGateway(client=chat_llm_client)
        elif gateway is not None:
            # If an explicit mock/custom gateway was passed, reuse it for tests
            self.chat_gateway = self.gateway
        elif llm_client is not None and not isinstance(self.llm_client, GroqLLMClient):
            # If an explicit mock/custom llm_client was passed, reuse the orchestrator gateway
            self.chat_gateway = self.gateway
        else:
            # Dedicated chat client for general conversation targeting qwen/qwen3.8-27b
            self.chat_client: LLMClientProtocol = GroqLLMClient(model=self.chat_model)
            self.chat_gateway = LLMGateway(client=self.chat_client)

        self.classifier = classifier or QueryClassifier()
        self.citation_verifier = citation_verifier or CitationVerifier()
        self.answer_generator = answer_generator or AnswerGenerator(llm_client=self.llm_client)

    @property
    def source(self) -> str:
        """Identify LLM service source ('real' or 'stand-in')."""
        return getattr(self.llm_client, "source", "unknown")

    async def run_pipeline(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> PipelineResult:
        """Run the end-to-end pipeline with intent routing and verification loops.

        Routes:
        1. DIRECT_LLM: Chit-chat, programming, general queries (bypasses retriever and verifier).
        2. EXTERNAL_WEB: Live web navigation via fetch_web_page and LLM synthesis.
        3. CORPUS_RAG: Standard 4-step RAG pipeline with verification and retry looping.

        Args:
            query: User input prompt or question.
            top_k: Override for number of candidate chunks in RAG retrieval.
            max_tokens: Maximum output tokens for LLM generation.
            temperature: LLM sampling temperature.

        Returns:
            PipelineResult containing answer, chunks, verification state, and telemetry.
        """
        t0 = time.perf_counter()
        steps: List[str] = ["step_1_analyze"]

        # =============================================================
        # STEP 1 (Analyze): Query Classification & Route Selection
        # =============================================================
        classification = self.classifier.classify(query)

        # -------------------------------------------------------------
        # ROUTE 1: DIRECT_LLM (Bypass retrieval and citation verification)
        # -------------------------------------------------------------
        if classification.intent == QueryIntent.DIRECT_LLM:
            steps.append("direct_llm")
            direct_messages = [
                {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ]
            comp: CompletionResult = await self.chat_gateway.complete_stream(
                messages=direct_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            end_to_end_ms = (time.perf_counter() - t0) * 1000.0
            return PipelineResult(
                answer=comp.content,
                chunks=[],
                verification_state=VerificationResult(
                    is_valid=True,
                    is_refusal=False,
                    valid_citations=[],
                    invalid_citations=[],
                    error_message=None,
                ),
                loop_count=0,
                telemetry={
                    "first_token_ms": round(comp.first_token_ms, 2),
                    "total_ms": round(comp.total_ms, 2),
                    "tokens_in": comp.tokens_in,
                    "tokens_out": comp.tokens_out,
                    "model": getattr(self.chat_gateway, "model", self.chat_model),
                    "source": getattr(self.chat_gateway, "source", self.source),
                    "retrieval_ms": 0.0,
                    "end_to_end_ms": round(end_to_end_ms, 2),
                },
                classification=classification,
                steps_executed=steps,
            )

        # -------------------------------------------------------------
        # ROUTE 2: EXTERNAL_WEB (Fetch web page context & synthesize)
        # -------------------------------------------------------------
        if classification.intent == QueryIntent.EXTERNAL_WEB:
            steps.append("external_web_fetch")
            url = classification.external_url
            if not url:
                url_match = re.search(r"https?://[^\s<>\"']+|www\.[^\s<>\"']+", query, re.IGNORECASE)
                url = url_match.group(0) if url_match else "https://en.wikipedia.org"
                if url.lower().startswith("www."):
                    url = "https://" + url

            from tools.web_fetcher import fetch_web_page

            t_web_0 = time.perf_counter()
            web_text = await fetch_web_page(url)
            web_fetch_ms = (time.perf_counter() - t_web_0) * 1000.0

            steps.append("external_web_synthesize")
            web_messages = [
                {"role": "system", "content": WEB_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Retrieved Web Context from {url}:\n"
                        f"==================================================\n"
                        f"{web_text}\n"
                        f"==================================================\n\n"
                        f"User Question: {query}\n"
                    ),
                },
            ]
            comp: CompletionResult = await self.gateway.complete_stream(
                messages=web_messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            end_to_end_ms = (time.perf_counter() - t0) * 1000.0
            return PipelineResult(
                answer=comp.content,
                chunks=[],
                verification_state=VerificationResult(
                    is_valid=True,
                    is_refusal=False,
                    valid_citations=[],
                    invalid_citations=[],
                    error_message=None,
                ),
                loop_count=0,
                telemetry={
                    "first_token_ms": round(comp.first_token_ms, 2),
                    "total_ms": round(comp.total_ms, 2),
                    "tokens_in": comp.tokens_in,
                    "tokens_out": comp.tokens_out,
                    "source": self.source,
                    "retrieval_ms": 0.0,
                    "web_fetch_ms": round(web_fetch_ms, 2),
                    "web_url": url,
                    "mcp_server": "playwright",
                    "end_to_end_ms": round(end_to_end_ms, 2),
                },
                classification=classification,
                steps_executed=steps,
            )

        # -------------------------------------------------------------
        # ROUTE 3: CORPUS_RAG (Full 4-Step Pipeline with Verification Loop)
        # -------------------------------------------------------------
        if not classification.is_in_scope:
            steps.append("step_1_refusal")
            end_to_end_ms = (time.perf_counter() - t0) * 1000.0
            return PipelineResult(
                answer=SAFE_OUT_OF_SCOPE_REFUSAL,
                chunks=[],
                verification_state=VerificationResult(
                    is_valid=True,
                    is_refusal=True,
                    valid_citations=[],
                    invalid_citations=[],
                    error_message=None,
                ),
                loop_count=0,
                telemetry={
                    "first_token_ms": 0.0,
                    "total_ms": 0.0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "source": self.source,
                    "retrieval_ms": 0.0,
                    "end_to_end_ms": round(end_to_end_ms, 2),
                },
                classification=classification,
                steps_executed=steps,
            )

        # STEP 2 (Tool Call): Retrieve Candidate Chunks
        steps.append("step_2_retrieve")
        t_ret_0 = time.perf_counter()

        effective_k = top_k or self.retriever.top_k
        if classification.target_papers and len(classification.target_papers) > 1:
            chunks = []
            per_paper_k = max(2, effective_k // len(classification.target_papers))
            for paper_id in classification.target_papers:
                paper_chunks = self.retriever.retrieve(
                    query=query,
                    top_k=per_paper_k,
                    filter_paper=paper_id,
                )
                chunks.extend(paper_chunks)
        else:
            chunks = self.retriever.retrieve(query=query, top_k=top_k)

        retrieval_ms = (time.perf_counter() - t_ret_0) * 1000.0

        cumulative_tokens_in = 0
        cumulative_tokens_out = 0

        # STEP 3 (Draft): Initial Answer Generation (Attempt 0)
        steps.append("step_3_draft")
        gen_0: GenerationResult = await self.answer_generator.generate(
            query=query,
            chunks=chunks,
            feedback=None,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        cumulative_tokens_in += gen_0.tokens_in
        cumulative_tokens_out += gen_0.tokens_out
        last_ttft_ms = gen_0.first_token_ms
        last_gen_total_ms = gen_0.total_ms

        # STEP 4 (Verify & Loop/Branch): Citation Verification (Attempt 0)
        steps.append("step_4_verify")
        ver_0 = self.citation_verifier.verify(gen_0.content, chunks)

        if ver_0.is_valid or ver_0.is_refusal:
            steps.append("accepted")
            end_to_end_ms = (time.perf_counter() - t0) * 1000.0
            return PipelineResult(
                answer=gen_0.content,
                chunks=chunks,
                verification_state=ver_0,
                loop_count=0,
                telemetry={
                    "first_token_ms": round(last_ttft_ms, 2),
                    "total_ms": round(last_gen_total_ms, 2),
                    "tokens_in": cumulative_tokens_in,
                    "tokens_out": cumulative_tokens_out,
                    "source": self.source,
                    "retrieval_ms": round(retrieval_ms, 2),
                    "end_to_end_ms": round(end_to_end_ms, 2),
                },
                classification=classification,
                steps_executed=steps,
            )

        # LOOP RULE: If invalid and loop_count == 0, retry Step 3 with feedback
        logger.warning(
            "Citation verification failed on attempt 0 (%s). Looping back to Step 3 with feedback.",
            ver_0.error_message,
        )
        steps.append("step_3_draft_retry")
        feedback_prompt = f"Fix citations to match only provided pages. Issue: {ver_0.error_message}"
        gen_1: GenerationResult = await self.answer_generator.generate(
            query=query,
            chunks=chunks,
            feedback=feedback_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        cumulative_tokens_in += gen_1.tokens_in
        cumulative_tokens_out += gen_1.tokens_out
        last_ttft_ms = gen_1.first_token_ms
        last_gen_total_ms = gen_1.total_ms

        steps.append("step_4_verify_retry")
        ver_1 = self.citation_verifier.verify(gen_1.content, chunks)

        if ver_1.is_valid or ver_1.is_refusal:
            steps.append("accepted_on_retry")
            end_to_end_ms = (time.perf_counter() - t0) * 1000.0
            return PipelineResult(
                answer=gen_1.content,
                chunks=chunks,
                verification_state=ver_1,
                loop_count=1,
                telemetry={
                    "first_token_ms": round(last_ttft_ms, 2),
                    "total_ms": round(last_gen_total_ms, 2),
                    "tokens_in": cumulative_tokens_in,
                    "tokens_out": cumulative_tokens_out,
                    "source": self.source,
                    "retrieval_ms": round(retrieval_ms, 2),
                    "end_to_end_ms": round(end_to_end_ms, 2),
                },
                classification=classification,
                steps_executed=steps,
            )

        # BRANCH RULE: Double failure branches to safe refusal
        logger.error(
            "Citation verification failed after 1 retry (%s). Branching to safe refusal.",
            ver_1.error_message,
        )
        steps.append("branch_safe_refusal")
        end_to_end_ms = (time.perf_counter() - t0) * 1000.0
        return PipelineResult(
            answer=SAFE_VALIDATION_FAILURE_REFUSAL,
            chunks=chunks,
            verification_state=ver_1,
            loop_count=1,
            telemetry={
                "first_token_ms": round(last_ttft_ms, 2),
                "total_ms": round(last_gen_total_ms, 2),
                "tokens_in": cumulative_tokens_in,
                "tokens_out": cumulative_tokens_out,
                "source": self.source,
                "retrieval_ms": round(retrieval_ms, 2),
                "end_to_end_ms": round(end_to_end_ms, 2),
            },
            classification=classification,
            steps_executed=steps,
        )

    async def execute(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> PipelineResult:
        """Execute the pipeline (alias for run_pipeline for backwards compatibility)."""
        return await self.run_pipeline(
            query=query,
            top_k=top_k,
            max_tokens=max_tokens,
            temperature=temperature,
        )


# Alias for Relay Assignment Brief naming
RelayPipelineOrchestrator = RAGOrchestrator
