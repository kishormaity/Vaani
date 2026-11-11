"""End-to-End RAG Pipeline for The Relay.

Combines FAISS/CrossEncoder retrieval with citation-grounded Groq LLM generation.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

from llm import CompletionResult, GroqLLMClient, LLMClientProtocol
from rag.prompt_templates import SYSTEM_PROMPT, format_context_prompt
from retrieval import RetrievedChunk, Retriever

logger = logging.getLogger(__name__)


@dataclass
class RAGResponse:
    """Complete response from the RAG pipeline with answer, citations, and latency telemetry."""

    query: str
    answer: str
    setting_name: str
    retrieved_chunks: List[RetrievedChunk]
    citations: List[str]
    retrieval_ms: float
    generation_ttft_ms: float
    generation_total_ms: float
    end_to_end_ms: float
    tokens_in: int
    tokens_out: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query": self.query,
            "answer": self.answer,
            "setting_name": self.setting_name,
            "citations": self.citations,
            "latency": {
                "retrieval_ms": round(self.retrieval_ms, 2),
                "generation_ttft_ms": round(self.generation_ttft_ms, 2),
                "generation_total_ms": round(self.generation_total_ms, 2),
                "end_to_end_ms": round(self.end_to_end_ms, 2),
            },
            "tokens": {
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
            },
            "retrieved_chunks": [c.to_dict() for c in self.retrieved_chunks],
        }


class RAGPipeline:
    """Coordinates retrieval and citation-grounded LLM generation."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        llm_client: Optional[LLMClientProtocol] = None,
        setting: str = "Setting A",
    ) -> None:
        self.setting_name = setting
        self.retriever = retriever or Retriever(setting=setting)
        self.llm_client = llm_client or GroqLLMClient()

    async def answer_query(
        self,
        query: str,
        top_k: Optional[int] = None,
        max_tokens: int = 500,
    ) -> RAGResponse:
        """Process a query through the full RAG pipeline with citation grounding.
        
        Args:
            query: User question about the research papers.
            top_k: Override for number of chunks to pass to LLM context.
            max_tokens: Maximum completion tokens for the LLM.
            
        Returns:
            RAGResponse object containing answer, citations, and telemetry.
        """
        t0 = time.perf_counter()

        # Step 1: Retrieval (Dense or Dense + CrossEncoder)
        t_ret_0 = time.perf_counter()
        chunks = self.retriever.retrieve(query=query, top_k=top_k)
        retrieval_ms = (time.perf_counter() - t_ret_0) * 1000.0

        # Step 2: Prompt Formulation
        user_prompt = format_context_prompt(query, chunks)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # Step 3: LLM Generation
        generation_res = await self.llm_client.complete_stream(
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.0,
        )

        # Step 4: Extract and Verify Citations
        citation_pattern = re.compile(r"\[arXiv:[^\]]+\]")
        found_citations = list(set(citation_pattern.findall(generation_res.content)))
        if not found_citations:
            # Fallback to the citations of the top retrieved chunks
            found_citations = [c.citation for c in chunks]

        end_to_end_ms = (time.perf_counter() - t0) * 1000.0

        return RAGResponse(
            query=query,
            answer=generation_res.content,
            setting_name=self.setting_name,
            retrieved_chunks=chunks,
            citations=sorted(found_citations),
            retrieval_ms=retrieval_ms,
            generation_ttft_ms=generation_res.first_token_ms,
            generation_total_ms=generation_res.total_ms,
            end_to_end_ms=end_to_end_ms,
            tokens_in=generation_res.tokens_in,
            tokens_out=generation_res.tokens_out,
        )
