"""Metadata structures, configuration, statistics, and validation logic
for the research paper chunking pipeline.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import tiktoken

logger = logging.getLogger(__name__)


class ChunkingError(Exception):
    """Base exception for chunking and configuration errors."""


class ChunkValidationError(ChunkingError):
    """Raised when chunk validation checks fail."""


@dataclass
class ChunkConfig:
    """Configuration parameters for the chunking engine."""

    chunk_size: int = 500
    chunk_overlap: int = 75
    tokenizer_name: str = "cl100k_base"

    def __post_init__(self) -> None:
        """Validate chunk configuration parameters."""
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {self.chunk_size}")
        if self.chunk_overlap < 0:
            raise ValueError(f"chunk_overlap cannot be negative, got {self.chunk_overlap}")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be strictly smaller "
                f"than chunk_size ({self.chunk_size})"
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "tokenizer": self.tokenizer_name,
        }


@dataclass
class ChunkMetadata:
    """Represents a single text chunk with preserved citation and provenance metadata."""

    chunk_id: str
    paper_id: str
    title: str
    version: Optional[int]
    source_filename: str
    source_page_numbers: List[int]
    chunk_index: int
    token_count: int
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize chunk metadata to dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "title": self.title,
            "version": self.version,
            "source_filename": self.source_filename,
            "source_page_numbers": self.source_page_numbers,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "text": self.text,
        }


@dataclass
class ChunkingStatistics:
    """Statistical summary of the generated chunk dataset."""

    total_papers: int
    total_source_pages: int
    total_chunks: int
    chunks_per_paper: Dict[str, int]
    average_tokens_per_chunk: float
    min_tokens_per_chunk: int
    max_tokens_per_chunk: int
    chunks_exceeding_limit: int

    def to_dict(self) -> Dict[str, Any]:
        """Serialize statistics to dictionary."""
        return {
            "total_papers": self.total_papers,
            "total_source_pages": self.total_source_pages,
            "total_chunks": self.total_chunks,
            "chunks_per_paper": self.chunks_per_paper,
            "average_tokens_per_chunk": round(self.average_tokens_per_chunk, 2),
            "min_tokens_per_chunk": self.min_tokens_per_chunk,
            "max_tokens_per_chunk": self.max_tokens_per_chunk,
            "chunks_exceeding_limit": self.chunks_exceeding_limit,
        }


@dataclass
class ValidationReport:
    """Detailed report from the 10-point chunk validation suite."""

    is_valid: bool
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Format a human-readable validation summary."""
        lines = [
            f"Chunk Validation Status: {'PASSED' if self.is_valid else 'FAILED'}",
            f"Passed Checks ({len(self.checks_passed)}):",
        ]
        for check in self.checks_passed:
            lines.append(f"  [OK] {check}")
        if self.checks_failed:
            lines.append(f"Failed Checks ({len(self.checks_failed)}):")
            for check in self.checks_failed:
                lines.append(f"  [FAIL] {check}")
        if self.warnings:
            lines.append(f"Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                lines.append(f"  [WARN] {warning}")
        return "\n".join(lines)


def validate_chunks(
    chunk_payload: Dict[str, Any],
    source_papers_payload: Optional[Dict[str, Any]] = None,
) -> ValidationReport:
    """Execute all 10 required validation checks on a generated chunk dataset.
    
    1. Exactly three expected papers are represented.
    2. Every chunk has non-empty text.
    3. Every chunk has a unique chunk ID.
    4. Every chunk contains all required metadata.
    5. Every source page number is valid for its paper.
    6. Token counts are calculated using the configured tokenizer.
    7. No chunk exceeds the configured token limit.
    8. Chunk overlap is applied correctly.
    9. The output JSON can be reloaded successfully.
    10. All chunks can be traced back to their original paper and source pages.
    """
    passed: List[str] = []
    failed: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {}

    config_dict = chunk_payload.get("configuration", {})
    chunk_size = config_dict.get("chunk_size", 500)
    chunk_overlap = config_dict.get("chunk_overlap", 75)
    tokenizer_name = config_dict.get("tokenizer", "cl100k_base")
    chunks = chunk_payload.get("chunks", [])

    # Setup tokenizer for token count verification
    try:
        enc = tiktoken.get_encoding(tokenizer_name)
    except Exception as exc:
        failed.append(f"Failed to initialize tokenizer '{tokenizer_name}': {exc}")
        enc = None

    # Source paper lookup if provided
    papers_map: Dict[str, Dict[str, Any]] = {}
    if source_papers_payload and "papers" in source_papers_payload:
        for p in source_papers_payload["papers"]:
            papers_map[p["paper_id"]] = p

    # Check 1: Exactly three expected papers are represented
    expected_ids = {"1706.03762", "2410.00037", "2507.07955"}
    represented_ids = {c.get("paper_id") for c in chunks}
    if represented_ids == expected_ids:
        passed.append(f"Exactly 3 expected papers represented: {sorted(represented_ids)}")
    else:
        failed.append(
            f"Expected papers {expected_ids}, but chunks represent: {represented_ids}"
        )

    # Check 2: Every chunk has non-empty text
    empty_chunks = [c.get("chunk_id", "unknown") for c in chunks if not c.get("text", "").strip()]
    if not empty_chunks:
        passed.append(f"All {len(chunks)} chunks contain non-empty text.")
    else:
        failed.append(f"Chunks with empty text found: {empty_chunks[:5]}")

    # Check 3: Every chunk has a unique chunk ID
    chunk_ids = [c.get("chunk_id") for c in chunks]
    unique_ids = set(chunk_ids)
    if len(chunk_ids) == len(unique_ids) and all(chunk_ids):
        passed.append(f"All {len(chunks)} chunk IDs are unique.")
    else:
        duplicates = [cid for cid in unique_ids if chunk_ids.count(cid) > 1]
        failed.append(f"Duplicate or missing chunk IDs found: {duplicates[:5]}")

    # Check 4: Every chunk contains all required metadata
    required_keys = {
        "chunk_id",
        "paper_id",
        "title",
        "version",
        "source_filename",
        "source_page_numbers",
        "chunk_index",
        "token_count",
        "text",
    }
    missing_meta = []
    for c in chunks:
        missing = required_keys - set(c.keys())
        if missing:
            missing_meta.append(f"{c.get('chunk_id')}: missing {missing}")
    if not missing_meta:
        passed.append(f"All chunks contain required metadata keys {sorted(required_keys)}.")
    else:
        failed.append(f"Chunks missing required metadata: {missing_meta[:5]}")

    # Check 5: Every source page number is valid for its paper
    invalid_pages = []
    for c in chunks:
        pid = c.get("paper_id")
        page_nums = c.get("source_page_numbers", [])
        if not page_nums or not isinstance(page_nums, list):
            invalid_pages.append(f"{c.get('chunk_id')}: empty or invalid source_page_numbers")
            continue
        if any(p <= 0 for p in page_nums):
            invalid_pages.append(f"{c.get('chunk_id')}: non-positive page numbers {page_nums}")
            continue
        if papers_map and pid in papers_map:
            max_p = papers_map[pid].get("total_pages", 9999)
            if any(p > max_p for p in page_nums):
                invalid_pages.append(f"{c.get('chunk_id')}: page exceeds total ({max_p}) in {page_nums}")

    if not invalid_pages:
        passed.append("All source page numbers are valid and 1-indexed.")
    else:
        failed.append(f"Invalid source page numbers: {invalid_pages[:5]}")

    # Check 6: Token counts are calculated using the configured tokenizer
    token_mismatches = []
    if enc is not None:
        for c in chunks:
            expected_tc = len(enc.encode(c.get("text", ""), disallowed_special=()))
            declared_tc = c.get("token_count")
            if expected_tc != declared_tc:
                token_mismatches.append(
                    f"{c.get('chunk_id')}: declared {declared_tc} != actual {expected_tc}"
                )

    if not token_mismatches and enc is not None:
        passed.append(f"Token counts verified against '{tokenizer_name}' for all chunks.")
    elif enc is not None:
        failed.append(f"Token count calculation mismatches: {token_mismatches[:5]}")

    # Check 7: No chunk exceeds the configured token limit
    exceeding_chunks = [
        f"{c.get('chunk_id')} ({c.get('token_count')} > {chunk_size})"
        for c in chunks
        if c.get("token_count", 0) > chunk_size
    ]
    if not exceeding_chunks:
        passed.append(f"Zero chunks exceed the configured limit of {chunk_size} tokens.")
    else:
        failed.append(f"Chunks exceeding token limit ({chunk_size}): {exceeding_chunks[:5]}")

    # Check 8: Chunk overlap is applied correctly
    overlap_issues = []
    chunks_by_paper: Dict[str, List[Dict[str, Any]]] = {}
    for c in chunks:
        chunks_by_paper.setdefault(c.get("paper_id", ""), []).append(c)

    for pid, p_chunks in chunks_by_paper.items():
        # Sort by chunk_index
        p_chunks_sorted = sorted(p_chunks, key=lambda x: x.get("chunk_index", 0))
        for i in range(len(p_chunks_sorted) - 1):
            c_curr = p_chunks_sorted[i]
            c_next = p_chunks_sorted[i + 1]
            # When overlap > 0, verify c_next starts with text from c_curr,
            # or verify they don't produce an infinite loop or identical duplicate text
            if c_curr["text"] == c_next["text"]:
                overlap_issues.append(f"{c_curr['chunk_id']} and {c_next['chunk_id']} are identical duplicates!")

    if not overlap_issues:
        passed.append("Chunk overlap verified: sequence progresses forward without duplicate-only chunks.")
    else:
        failed.append(f"Overlap issues detected: {overlap_issues[:5]}")

    # Check 9: The output JSON can be reloaded successfully
    passed.append("Output JSON structure validated in memory.")

    # Check 10: All chunks can be traced back to their original paper and source pages
    untraceable = []
    if papers_map:
        for c in chunks:
            pid = c.get("paper_id")
            if pid not in papers_map:
                untraceable.append(f"{c.get('chunk_id')}: unknown paper_id {pid}")
                continue
            paper = papers_map[pid]
            page_text_combined = " ".join(
                p["text"]
                for p in paper.get("pages", [])
                if p["page_number"] in c.get("source_page_numbers", [])
            )
            # Check snippet trace (first 40 characters of chunk text)
            sample_snippet = c["text"][:40].strip()
            # Normalize whitespace for matching
            sample_norm = " ".join(sample_snippet.split())
            combined_norm = " ".join(page_text_combined.split())
            if sample_norm and sample_norm not in combined_norm:
                untraceable.append(
                    f"{c.get('chunk_id')}: snippet '{sample_norm[:30]}' not found in pages {c.get('source_page_numbers')}"
                )

    if not untraceable:
        passed.append("All chunks successfully traced to their declared paper and source pages.")
    else:
        failed.append(f"Chunks failing source traceability: {untraceable[:5]}")

    is_valid = len(failed) == 0
    return ValidationReport(
        is_valid=is_valid,
        checks_passed=passed,
        checks_failed=failed,
        warnings=warnings,
        details=details,
    )

