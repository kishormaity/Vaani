"""Backwards-compatibility shim re-exporting from citation_verifier."""

from pipeline.citation_verifier import (
    VERIFIED_PAPER_IDS,
    CitationValidator,
    CitationVerifier,
    ValidationResult,
    VerificationResult,
)

__all__ = [
    "CitationValidator",
    "CitationVerifier",
    "ValidationResult",
    "VerificationResult",
    "VERIFIED_PAPER_IDS",
]
