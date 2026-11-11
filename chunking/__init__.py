"""Text chunking package for The Relay.

Implements configurable, paragraph-aware text chunking with full citation metadata preservation.
"""

from chunking.chunker import ParagraphAwareChunker, Tokenizer
from chunking.metadata import (
    ChunkConfig,
    ChunkMetadata,
    ChunkValidationError,
    ChunkingError,
    ChunkingStatistics,
    ValidationReport,
    validate_chunks,
)

__all__ = [
    "ParagraphAwareChunker",
    "Tokenizer",
    "ChunkConfig",
    "ChunkMetadata",
    "ChunkingStatistics",
    "ValidationReport",
    "ChunkingError",
    "ChunkValidationError",
    "validate_chunks",
]

