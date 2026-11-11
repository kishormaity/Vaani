"""Retrieval and Re-ranking package for The Relay.

Implements modular dense vector retrieval and CrossEncoder re-ranking
matching Table 3 experimental presets.
"""

from retrieval.reranker import CrossEncoderReranker
from retrieval.retriever import (
    PRESET_CONFIGURATIONS,
    RetrievedChunk,
    Retriever,
    format_citation,
)

__all__ = [
    "CrossEncoderReranker",
    "Retriever",
    "RetrievedChunk",
    "PRESET_CONFIGURATIONS",
    "format_citation",
]

