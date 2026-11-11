"""Embeddings package for The Relay.

Provides dense vector generation using Sentence Transformers.
"""

from embeddings.embedder import Embedder
from embeddings.metadata import (
    ChunkVectorMetadata,
    EmbeddingConfig,
    EmbeddingValidationReport,
    validate_embeddings,
)

__all__ = [
    "Embedder",
    "EmbeddingConfig",
    "ChunkVectorMetadata",
    "EmbeddingValidationReport",
    "validate_embeddings",
]

