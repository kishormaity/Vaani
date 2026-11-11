"""Metadata models, configuration, and validation for chunk embeddings.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingConfig:
    """Configuration for dense vector embedding generation."""

    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    normalize_embeddings: bool = True
    device: Optional[str] = None
    expected_dimension: int = 384

    def __post_init__(self) -> None:
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.expected_dimension <= 0:
            raise ValueError(f"expected_dimension must be positive, got {self.expected_dimension}")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "batch_size": self.batch_size,
            "normalize_embeddings": self.normalize_embeddings,
            "expected_dimension": self.expected_dimension,
            "device": self.device or "auto",
        }


@dataclass
class ChunkVectorMetadata:
    """Metadata linking a vector index position to its source chunk provenance."""

    vector_index: int
    chunk_id: str
    paper_id: str
    title: str
    version: Optional[int]
    source_filename: str
    source_page_numbers: List[int]
    chunk_index: int
    token_count: int
    embedding_model: str
    embedding_dimension: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector_index": self.vector_index,
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "title": self.title,
            "version": self.version,
            "source_filename": self.source_filename,
            "source_page_numbers": self.source_page_numbers,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
        }


@dataclass
class EmbeddingValidationReport:
    """Detailed report for embedding validation."""

    is_valid: bool
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            f"Embedding Validation Status: {'PASSED' if self.is_valid else 'FAILED'}",
            f"Passed Checks ({len(self.checks_passed)}):",
        ]
        for c in self.checks_passed:
            lines.append(f"  [OK] {c}")
        if self.checks_failed:
            lines.append(f"Failed Checks ({len(self.checks_failed)}):")
            for c in self.checks_failed:
                lines.append(f"  [FAIL] {c}")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for w in self.warnings:
                lines.append(f"  [WARN] {w}")
        return "\n".join(lines)


def validate_embeddings(
    embeddings: np.ndarray,
    metadata_list: List[Dict[str, Any]],
    expected_count: int = 254,
    expected_dim: int = 384,
) -> EmbeddingValidationReport:
    """Validate numerical properties and metadata alignment of generated embeddings."""
    passed: List[str] = []
    failed: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {}

    # Check 1: Array type and shape
    if not isinstance(embeddings, np.ndarray):
        failed.append(f"Embeddings object is not a numpy ndarray (type: {type(embeddings)})")
        return EmbeddingValidationReport(is_valid=False, checks_failed=failed)

    if embeddings.ndim != 2:
        failed.append(f"Embeddings array must be 2D, got shape {embeddings.shape}")
        return EmbeddingValidationReport(is_valid=False, checks_failed=failed)

    num_vectors, dim = embeddings.shape
    if num_vectors == expected_count:
        passed.append(f"Embedding count matches expected count ({num_vectors} == {expected_count}).")
    else:
        failed.append(f"Embedding count mismatch: got {num_vectors}, expected {expected_count}.")

    # Check 2: Dimension
    if dim == expected_dim:
        passed.append(f"Embedding vector dimension matches expected ({dim} == {expected_dim}).")
    else:
        failed.append(f"Embedding dimension mismatch: got {dim}, expected {expected_dim}.")

    # Check 3: No NaN or Infinite values
    has_nan = np.isnan(embeddings).any()
    has_inf = np.isinf(embeddings).any()
    if not has_nan and not has_inf:
        passed.append("No NaN or infinite values exist in the embedding matrix.")
    else:
        failed.append(f"Numerical errors in embeddings: NaN={has_nan}, Inf={has_inf}.")

    # Check 4: L2 Normalization check (unit norm ~ 1.0)
    norms = np.linalg.norm(embeddings, axis=1)
    norm_diffs = np.abs(norms - 1.0)
    max_norm_diff = float(np.max(norm_diffs))
    if max_norm_diff < 1e-4:
        passed.append(f"All vectors are L2-normalized to unit length (max deviation: {max_norm_diff:.2e}).")
    else:
        failed.append(f"Vectors not normalized to unit length: max deviation = {max_norm_diff:.4f}.")

    # Check 5: Metadata alignment
    if len(metadata_list) == num_vectors:
        passed.append(f"Metadata list length matches vector count ({len(metadata_list)} == {num_vectors}).")
    else:
        failed.append(f"Metadata count ({len(metadata_list)}) does not match vector count ({num_vectors}).")

    # Check 6: Unique chunk IDs & sequential vector indices
    chunk_ids = [m.get("chunk_id") for m in metadata_list]
    unique_ids = set(chunk_ids)
    if len(chunk_ids) == len(unique_ids) and all(chunk_ids):
        passed.append(f"All {len(chunk_ids)} chunk IDs in embedding metadata are unique.")
    else:
        failed.append("Duplicate or empty chunk IDs found in embedding metadata.")

    index_align_errs = []
    for i, m in enumerate(metadata_list):
        if m.get("vector_index") != i:
            index_align_errs.append(f"Row {i} has vector_index={m.get('vector_index')}")

    if not index_align_errs:
        passed.append("Every FAISS/matrix row index aligns deterministically with metadata 'vector_index'.")
    else:
        failed.append(f"Index alignment errors: {index_align_errs[:5]}")

    details["num_vectors"] = num_vectors
    details["dimension"] = dim
    details["max_norm_deviation"] = max_norm_diff

    return EmbeddingValidationReport(
        is_valid=len(failed) == 0,
        checks_passed=passed,
        checks_failed=failed,
        warnings=warnings,
        details=details,
    )

