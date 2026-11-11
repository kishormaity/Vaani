"""Metadata models, expected paper specifications, and validation utilities
for the research paper extraction module.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Canonical specifications of the three expected papers
EXPECTED_PAPERS: Dict[str, Dict[str, Any]] = {
    "1706.03762": {
        "paper_id": "1706.03762",
        "title": "Attention Is All You Need",
        "version": 7,
        "keywords": ["attention is all you need", "vaswani"],
    },
    "2410.00037": {
        "paper_id": "2410.00037",
        "title": "Moshi: A Speech-Text Foundation Model for Real-Time Dialogue",
        "version": 1,
        "keywords": ["moshi", "speech-text foundation model", "real-time dialogue"],
    },
    "2507.07955": {
        "paper_id": "2507.07955",
        "title": "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling",
        "version": 2,
        "keywords": ["dynamic chunking", "hierarchical sequence modeling"],
    },
}

# Regex to capture arXiv ID and optional version from text or metadata strings
# Matches: "arXiv:1706.03762v7", "arxiv: 2410.00037v1", "1706.03762", etc.
ARXIV_REGEX = re.compile(
    r"(?:arxiv[:\s]*)?([0-9]{4}\.[0-9]{4,5})(?:v([0-9]+))?",
    re.IGNORECASE,
)


class ExtractionError(Exception):
    """Base exception for PDF extraction and validation errors."""


class VersionMismatchError(ExtractionError):
    """Raised when an extracted paper version does not match the expected version."""


class PaperValidationError(ExtractionError):
    """Raised when validation criteria are violated."""


@dataclass
class PageData:
    """Represents the extracted text and metadata of a single page."""

    page_number: int  # 1-indexed
    paper_id: str
    title: str
    version: Optional[int]
    text: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert page data to a dictionary."""
        return {
            "page_number": self.page_number,
            "paper_id": self.paper_id,
            "title": self.title,
            "version": self.version,
            "text": self.text,
        }


@dataclass
class PaperMetadata:
    """Represents extracted content and metadata for a complete research paper."""

    paper_id: str
    title: str
    version: Optional[int]
    filename: str
    total_pages: int
    extracted_pages: int
    empty_pages: List[int] = field(default_factory=list)
    status: str = "success"  # "success" | "partial" | "failed"
    pages: List[PageData] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert document metadata and pages to a dictionary matching required schema."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "version": self.version,
            "filename": self.filename,
            "total_pages": self.total_pages,
            "extracted_pages": self.extracted_pages,
            "empty_pages": self.empty_pages,
            "status": self.status,
            "pages": [page.to_dict() for page in self.pages],
        }


@dataclass
class ValidationReport:
    """Report detailing the results of validating extracted papers."""

    is_valid: bool
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Produce a human-readable summary of validation results."""
        lines = [
            f"Validation Status: {'PASSED' if self.is_valid else 'FAILED'}",
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


def extract_arxiv_from_text(text: str) -> Tuple[Optional[str], Optional[int]]:
    """Scan text for arXiv ID and version number.
    
    Returns (arxiv_id, version_int) if found, else (None, None).
    """
    if not text:
        return None, None

    match = ARXIV_REGEX.search(text)
    if match:
        arxiv_id = match.group(1)
        version = int(match.group(2)) if match.group(2) else None
        return arxiv_id, version

    return None, None


def verify_paper_identity(
    filename: str,
    first_pages_text: str,
    doc_metadata: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, Optional[int], List[str]]:
    """Verify paper identity using content and metadata rather than just filename.
    
    Args:
        filename: Original PDF filename.
        first_pages_text: Extracted text from initial pages (containing title/arXiv stamp).
        doc_metadata: PyMuPDF doc.metadata dictionary.
        
    Returns:
        (paper_id, title, version, warnings_list)
        
    Raises:
        VersionMismatchError: If the detected version differs from the expected version.
        ExtractionError: If paper identity cannot be resolved.
    """
    warnings: List[str] = []
    doc_metadata = doc_metadata or {}

    # 1. Look for arXiv identifier in the text (typically page 1 margin/footer)
    detected_id, detected_version = extract_arxiv_from_text(first_pages_text)

    # 2. Check metadata fields if text didn't yield an ID
    if not detected_id:
        for meta_val in doc_metadata.values():
            if isinstance(meta_val, str):
                d_id, d_ver = extract_arxiv_from_text(meta_val)
                if d_id:
                    detected_id = d_id
                    detected_version = detected_version or d_ver
                    break

    # 3. Match against expected papers by detected ID
    matched_expected: Optional[Dict[str, Any]] = None
    if detected_id and detected_id in EXPECTED_PAPERS:
        matched_expected = EXPECTED_PAPERS[detected_id]
    else:
        # Match by content keywords if ID wasn't explicitly found
        lowered_text = (first_pages_text + " " + doc_metadata.get("title", "")).lower()
        for exp_id, exp_info in EXPECTED_PAPERS.items():
            if any(kw in lowered_text for kw in exp_info["keywords"]):
                matched_expected = exp_info
                detected_id = exp_id
                warnings.append(
                    f"ArXiv ID not directly found in text for '{filename}'; "
                    f"matched to {exp_id} via content keywords."
                )
                break

    if not matched_expected:
        raise ExtractionError(
            f"Could not verify identity for file '{filename}'. "
            f"Detected arXiv ID '{detected_id}' does not match any expected paper."
        )

    canonical_title = matched_expected["title"]
    expected_version = matched_expected["version"]
    resolved_id = matched_expected["paper_id"]

    # 4. Resolve and verify version
    if detected_version is None:
        # Try extracting version from filename as fallback
        fn_match = re.search(r"v([0-9]+)", filename, re.IGNORECASE)
        if fn_match:
            detected_version = int(fn_match.group(1))
            warnings.append(
                f"Version not found in page text for '{filename}'; "
                f"inferred v{detected_version} from filename."
            )
        else:
            warnings.append(
                f"Version could not be detected for '{filename}'; "
                f"assuming expected version {expected_version}."
            )
            detected_version = expected_version

    # STRICT CHECK: report mismatch rather than silently accepting it
    if detected_version != expected_version:
        mismatch_msg = (
            f"Version mismatch for paper '{canonical_title}' ({resolved_id}): "
            f"expected version {expected_version}, but detected version {detected_version}."
        )
        logger.error(mismatch_msg)
        raise VersionMismatchError(mismatch_msg)

    return resolved_id, canonical_title, detected_version, warnings


def validate_extraction(
    papers: List[PaperMetadata],
    json_path: Optional[Path | str] = None,
) -> ValidationReport:
    """Validate extraction results against all 7 required validation criteria.
    
    1. Exactly three expected papers are present.
    2. Each PDF can be opened (status == 'success').
    3. Extracted text is not empty.
    4. Page numbers are sequential and correct (1-indexed).
    5. Each paper has the expected arXiv ID and version.
    6. The JSON file is valid and can be reloaded using Python.
    7. Extraction failures are clearly reported.
    """
    passed: List[str] = []
    failed: List[str] = []
    warnings: List[str] = []
    details: Dict[str, Any] = {}

    # Check 1: Exactly three expected papers are present
    expected_ids = set(EXPECTED_PAPERS.keys())
    extracted_ids = {p.paper_id for p in papers}
    if len(papers) == 3 and extracted_ids == expected_ids:
        passed.append(f"Exactly 3 expected papers present: {sorted(extracted_ids)}")
    else:
        failed.append(
            f"Expected exactly 3 papers {expected_ids}, got {len(papers)}: {extracted_ids}"
        )

    # Check 2: Each PDF was opened successfully
    open_failed = [p.filename for p in papers if p.status == "failed"]
    if not open_failed:
        passed.append(f"All {len(papers)} PDFs opened successfully.")
    else:
        failed.append(f"PDF open failed for files: {open_failed}")

    # Check 3: Extracted text is not empty
    empty_text_pages: List[str] = []
    for paper in papers:
        if paper.extracted_pages == 0:
            failed.append(f"Paper '{paper.filename}' has 0 extracted pages.")
        for page in paper.pages:
            if not page.text or not page.text.strip():
                empty_text_pages.append(f"{paper.filename}:page_{page.page_number}")

    if not empty_text_pages:
        passed.append("Extracted text is non-empty for all extracted pages.")
    else:
        failed.append(f"Empty text found on extracted pages: {empty_text_pages}")

    # Check 4: Page numbers are sequential and correct
    page_num_errors: List[str] = []
    for paper in papers:
        expected_seq = [
            p for p in range(1, paper.total_pages + 1) if p not in paper.empty_pages
        ]
        actual_seq = [p.page_number for p in paper.pages]
        if actual_seq != expected_seq:
            page_num_errors.append(
                f"{paper.filename}: expected {expected_seq[:5]}... got {actual_seq[:5]}..."
            )

    if not page_num_errors:
        passed.append("Page numbers are sequential, 1-indexed, and correct.")
    else:
        failed.append(f"Page numbering errors detected: {page_num_errors}")

    # Check 5: Expected arXiv ID and version match
    version_errors: List[str] = []
    for paper in papers:
        if paper.paper_id not in EXPECTED_PAPERS:
            version_errors.append(f"Unknown paper ID: {paper.paper_id}")
            continue
        exp_ver = EXPECTED_PAPERS[paper.paper_id]["version"]
        if paper.version != exp_ver:
            version_errors.append(
                f"{paper.paper_id} ('{paper.title}'): expected v{exp_ver}, got v{paper.version}"
            )

    if not version_errors:
        passed.append("All papers have verified expected arXiv IDs and versions.")
    else:
        failed.append(f"arXiv ID or version mismatches: {version_errors}")

    # Check 6: JSON file is valid and can be loaded again
    if json_path is not None:
        target_path = Path(json_path)
        if not target_path.exists():
            failed.append(f"Output JSON file does not exist: {target_path}")
        else:
            try:
                with open(target_path, "r", encoding="utf-8") as f:
                    reloaded = json.load(f)
                if "papers" in reloaded and isinstance(reloaded["papers"], list):
                    if len(reloaded["papers"]) == len(papers):
                        passed.append(
                            f"JSON file '{target_path.name}' is valid and reloaded successfully ({len(reloaded['papers'])} papers)."
                        )
                    else:
                        failed.append(
                            f"Reloaded JSON contains {len(reloaded['papers'])} papers, expected {len(papers)}."
                        )
                else:
                    failed.append("Reloaded JSON missing root 'papers' list.")
            except Exception as exc:
                failed.append(f"Failed to reload JSON file: {exc}")

    # Check 7: Extraction failures clearly reported
    total_errors = sum(len(p.errors) for p in papers)
    if total_errors > 0:
        for p in papers:
            for err in p.errors:
                failed.append(f"Extraction error reported for {p.filename}: {err}")
    else:
        passed.append("No unhandled extraction failures reported.")

    details["papers_summary"] = [
        {
            "filename": p.filename,
            "paper_id": p.paper_id,
            "title": p.title,
            "version": p.version,
            "total_pages": p.total_pages,
            "extracted_pages": p.extracted_pages,
            "empty_pages": p.empty_pages,
            "status": p.status,
        }
        for p in papers
    ]

    is_valid = len(failed) == 0
    return ValidationReport(
        is_valid=is_valid,
        checks_passed=passed,
        checks_failed=failed,
        warnings=warnings,
        details=details,
    )

