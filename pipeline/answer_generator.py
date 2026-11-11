"""Answer Generator for The Relay RAG pipeline.

Formats reference context, handles retry feedback prompting, and communicates
with LLMClientProtocol implementations (GroqLLMClient or StandinLLMService).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from llm.base import CompletionResult, LLMClientProtocol, RateLimitTelemetry
from retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are The Relay, an expert academic AI research assistant for question-answering over computer science research papers.

Guidelines:
1. Answer the question thoroughly, accurately, and concisely based ONLY on the provided context passages.
2. Every substantive factual claim must cite its source passage using the exact citation provided (e.g. [arXiv:1706.03762v7, p.5]).
3. If the provided passages do not contain sufficient information to answer the question, state clearly that the answer cannot be found in the provided research paper passages.
4. Do not invent or extrapolate facts beyond what is directly stated in the reference context."""


@dataclass
class GenerationResult:
    """Summary of LLM answer generation."""

    content: str
    first_token_ms: float
    total_ms: float
    tokens_in: int
    tokens_out: int
    telemetry: RateLimitTelemetry

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "first_token_ms": round(self.first_token_ms, 2),
            "total_ms": round(self.total_ms, 2),
            "tokens_in": self.tokens_in,
            "tokens_out": self.tokens_out,
            "telemetry": self.telemetry.raw_headers,
        }


class AnswerGenerator:
    """Coordinates context prompt formulation and LLM completion."""

    def __init__(self, llm_client: LLMClientProtocol) -> None:
        self.llm_client = llm_client

    def format_prompt(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        feedback: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        """Format retrieved context and question into system and user messages.

        Args:
            query: User question.
            chunks: List of retrieved context passages.
            feedback: Optional feedback string from previous failed citation validation.

        Returns:
            List of message dicts ready for LLM completion.
        """
        context_blocks: List[str] = []
        for i, chunk in enumerate(chunks, start=1):
            pages_str = ", ".join(str(p) for p in chunk.page_numbers)
            block = (
                f"--- Reference Passage [{i}] ---\n"
                f"Paper: {chunk.paper_title}\n"
                f"Paper ID: {chunk.paper_id}\n"
                f"Pages: [{pages_str}]\n"
                f"Required Citation Tag: {chunk.citation}\n"
                f"Content:\n{chunk.text.strip()}\n"
            )
            context_blocks.append(block)

        joined_context = "\n".join(context_blocks)

        user_content = (
            f"Context Passages:\n"
            f"==================================================\n"
            f"{joined_context}\n"
            f"==================================================\n\n"
            f"Question: {query}\n\n"
        )

        if feedback:
            user_content += (
                f"ATTENTION - CORRECTION FEEDBACK FROM PREVIOUS DRAFT:\n"
                f"{feedback}\n\n"
                f"Please rewrite your answer addressing the feedback above. "
                f"You MUST strictly cite ONLY the papers and page numbers available in the reference passages.\n\n"
            )

        user_content += (
            "Please provide a well-structured answer with inline academic citations "
            "pointing to the reference passages above."
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

    async def generate(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        feedback: Optional[str] = None,
        max_tokens: int = 500,
        temperature: float = 0.0,
    ) -> GenerationResult:
        """Execute LLM generation for the query and context.

        Args:
            query: User question.
            chunks: Retrieved context passages.
            feedback: Optional feedback from previous attempt.
            max_tokens: Maximum completion tokens.
            temperature: Sampling temperature.

        Returns:
            GenerationResult containing answer text and latency telemetry.
        """
        messages = self.format_prompt(query=query, chunks=chunks, feedback=feedback)

        completion: CompletionResult = await self.llm_client.complete_stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        return GenerationResult(
            content=completion.content,
            first_token_ms=completion.first_token_ms,
            total_ms=completion.total_ms,
            tokens_in=completion.tokens_in,
            tokens_out=completion.tokens_out,
            telemetry=completion.telemetry,
        )

