"""End-to-end RAG pipeline package for The Relay."""

from rag.pipeline import RAGPipeline, RAGResponse
from rag.prompt_templates import SYSTEM_PROMPT, format_context_prompt

__all__ = [
    "RAGPipeline",
    "RAGResponse",
    "SYSTEM_PROMPT",
    "format_context_prompt",
]

