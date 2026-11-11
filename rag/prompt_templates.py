"""Prompt templates and context formatting for The Relay RAG pipeline.
"""

from __future__ import annotations

from typing import List
from retrieval import RetrievedChunk

SYSTEM_PROMPT = """You are The Relay, an expert academic AI research assistant for question-answering over computer science research papers.

Guidelines:
1. Answer the question thoroughly, accurately, and concisely based ONLY on the provided context passages.
2. Every substantive factual claim must cite its source passage using the exact citation provided (e.g. [arXiv:1706.03762v7, p.5]).
3. If the provided passages do not contain sufficient information to answer the question, state clearly that the answer cannot be found in the provided research paper passages.
4. Do not invent or extrapolate facts beyond what is directly stated in the reference context."""


def format_context_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into a structured reference context for the LLM."""
    context_blocks = []
    for i, chunk in enumerate(chunks, start=1):
        block = (
            f"--- Reference Passage [{i}] ---\n"
            f"Paper: {chunk.paper_title}\n"
            f"Citation: {chunk.citation}\n"
            f"Chunk ID: {chunk.chunk_id}\n"
            f"Content:\n{chunk.text.strip()}\n"
        )
        context_blocks.append(block)

    joined_context = "\n".join(context_blocks)

    user_prompt = (
        f"Context Passages:\n"
        f"==================================================\n"
        f"{joined_context}\n"
        f"==================================================\n\n"
        f"Question: {query}\n\n"
        f"Please provide a well-structured answer with inline academic citations pointing to the reference passages above."
    )
    return user_prompt

