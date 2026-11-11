"""Data extraction package for The Relay.

Provides PyMuPDF-based text and metadata extraction from PDF research papers.
"""

from data_extraction.metadata import (
    EXPECTED_PAPERS,
    ExtractionError,
    PageData,
    PaperMetadata,
    PaperValidationError,
    ValidationReport,
    VersionMismatchError,
    validate_extraction,
    verify_paper_identity,
)
from data_extraction.pdf_extractor import PDFExtractor, clean_text

__all__ = [
    "PDFExtractor",
    "clean_text",
    "PageData",
    "PaperMetadata",
    "ValidationReport",
    "EXPECTED_PAPERS",
    "ExtractionError",
    "VersionMismatchError",
    "PaperValidationError",
    "verify_paper_identity",
    "validate_extraction",
]

