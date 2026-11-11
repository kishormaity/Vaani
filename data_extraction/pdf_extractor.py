"""PDF text extraction module using PyMuPDF (fitz).

Implements modular, robust extraction of page-by-page text from research papers,
preserving page numbering, document metadata, and paragraph structure while
handling errors gracefully.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import warnings

# Suppress the PyMuPDF 2.x deprecation warning for 'import fitz' while strictly
# adhering to the requirement: "Import it using: import fitz"
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, module="fitz")
    import fitz

from data_extraction.metadata import (
    EXPECTED_PAPERS,
    ExtractionError,
    PageData,
    PaperMetadata,
    VersionMismatchError,
    verify_paper_identity,
)

logger = logging.getLogger(__name__)


def clean_text(raw_text: str) -> str:
    """Clean extracted text by removing unnecessary whitespace without destroying paragraph structure.
    
    Operations:
    1. Normalizes carriage returns and form feeds into newlines.
    2. Replaces non-breaking spaces (\xa0) and removes zero-width spaces (\u200b).
    3. Trims trailing whitespace from each line and reduces multiple horizontal spaces.
    4. Preserves paragraph breaks while collapsing 3 or more consecutive newlines into 2.
    5. Strips outer leading and trailing whitespace.
    """
    if not raw_text:
        return ""

    # Normalize special characters and line endings
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n").replace("\x0c", "\n")
    text = text.replace("\xa0", " ").replace("\u200b", "")

    # Clean line by line
    cleaned_lines: List[str] = []
    for line in text.split("\n"):
        # Strip trailing whitespace and collapse consecutive horizontal spaces/tabs
        cleaned_line = " ".join(line.split())
        cleaned_lines.append(cleaned_line)

    joined = "\n".join(cleaned_lines)

    # Collapse excessive blank lines (3 or more newlines become 2 newlines for paragraph break)
    import re

    cleaned = re.sub(r"\n{3,}", "\n\n", joined)
    return cleaned.strip()


class PDFExtractor:
    """Modular PDF text extractor using PyMuPDF."""

    def __init__(self, expected_papers: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        """Initialize extractor with optional expected paper definitions."""
        self.expected_papers = expected_papers or EXPECTED_PAPERS

    def extract_paper(self, pdf_path: Union[str, Path]) -> PaperMetadata:
        """Extract text and metadata from a single PDF research paper.
        
        Args:
            pdf_path: Path to the PDF file.
            
        Returns:
            PaperMetadata containing document-level and page-level information.
        """
        path = Path(pdf_path)
        filename = path.name

        logger.info("PDF processing started: %s", filename)

        if not path.exists():
            error_msg = f"PDF file does not exist: {path}"
            logger.error(error_msg)
            return PaperMetadata(
                paper_id="unknown",
                title="unknown",
                version=None,
                filename=filename,
                total_pages=0,
                extracted_pages=0,
                empty_pages=[],
                status="failed",
                pages=[],
                errors=[error_msg],
            )

        doc = None
        try:
            doc = fitz.open(str(path))
        except Exception as exc:
            error_msg = f"Failed to open PDF '{filename}': {exc}"
            logger.error(error_msg, exc_info=True)
            return PaperMetadata(
                paper_id="unknown",
                title="unknown",
                version=None,
                filename=filename,
                total_pages=0,
                extracted_pages=0,
                empty_pages=[],
                status="failed",
                pages=[],
                errors=[error_msg],
            )

        total_pages = len(doc)
        logger.info("PDF '%s' opened successfully: %d total pages", filename, total_pages)

        # Inspect initial page(s) to verify identity from content and metadata
        first_pages_text = ""
        for i in range(min(total_pages, 3)):
            try:
                first_pages_text += doc[i].get_text("text") + "\n"
            except Exception as e:
                logger.warning("Could not read preview text from page %d of '%s': %s", i + 1, filename, e)

        # Verify paper identity against expected papers
        errors: List[str] = []
        try:
            paper_id, title, version, identity_warnings = verify_paper_identity(
                filename=filename,
                first_pages_text=first_pages_text,
                doc_metadata=doc.metadata,
            )
            for w in identity_warnings:
                logger.warning("[%s] %s", filename, w)
            logger.info(
                "Paper identity verified: ID='%s', Title='%s', Version=%s",
                paper_id,
                title,
                version,
            )
        except VersionMismatchError as vm_err:
            error_msg = str(vm_err)
            logger.error("[%s] %s", filename, error_msg)
            errors.append(error_msg)
            # Re-raise or record: per requirements, record failure and report mismatch
            paper_id = "unknown"
            title = filename
            version = None
        except Exception as id_err:
            error_msg = f"Identity verification error for '{filename}': {id_err}"
            logger.error(error_msg)
            errors.append(error_msg)
            paper_id = "unknown"
            title = filename
            version = None

        extracted_pages: List[PageData] = []
        empty_pages: List[int] = []

        try:
            for page_index in range(total_pages):
                # Always tie page_number to the physical PDF page index (1-indexed).
                # Skipping empty pages will never alter or decrement subsequent page numbers.
                page_number = page_index + 1
                try:
                    page = doc[page_index]
                    raw_text = page.get_text("text")
                    cleaned = clean_text(raw_text)

                    if not cleaned:
                        logger.warning(
                            "Page %d of '%s' is empty, skipping text extraction but recording in metadata",
                            page_number,
                            filename,
                        )
                        empty_pages.append(page_number)
                        continue

                    page_data = PageData(
                        page_number=page_number,  # Strictly preserves original physical PDF page number
                        paper_id=paper_id,
                        title=title,
                        version=version,
                        text=cleaned,
                    )
                    extracted_pages.append(page_data)
                    logger.info(
                        "Page %d/%d successfully extracted for '%s' (%d characters)",
                        page_number,
                        total_pages,
                        filename,
                        len(cleaned),
                    )

                except Exception as page_exc:
                    page_err = f"Failed to extract page {page_number} of '{filename}': {page_exc}"
                    logger.error(page_err, exc_info=True)
                    errors.append(page_err)

        finally:
            doc.close()

        # Determine extraction status
        if errors and not extracted_pages:
            status = "failed"
        elif errors or empty_pages:
            status = "partial" if errors else "success"
        else:
            status = "success"

        paper_meta = PaperMetadata(
            paper_id=paper_id,
            title=title,
            version=version,
            filename=filename,
            total_pages=total_pages,
            extracted_pages=len(extracted_pages),
            empty_pages=empty_pages,
            status=status,
            pages=extracted_pages,
            errors=errors,
        )

        logger.info(
            "Extraction completed for '%s': status=%s, extracted=%d/%d pages, empty=%d",
            filename,
            status,
            len(extracted_pages),
            total_pages,
            len(empty_pages),
        )

        return paper_meta

    def extract_directory(self, papers_dir: Union[str, Path]) -> List[PaperMetadata]:
        """Extract all PDF files within the given directory.
        
        Args:
            papers_dir: Directory containing research paper PDFs.
            
        Returns:
            List of PaperMetadata objects for all processed files.
        """
        dir_path = Path(papers_dir)
        if not dir_path.exists() or not dir_path.is_dir():
            raise FileNotFoundError(f"Papers directory not found: {dir_path}")

        pdf_files = sorted(dir_path.glob("*.pdf"))
        logger.info("Found %d PDF file(s) in %s", len(pdf_files), dir_path)

        results: List[PaperMetadata] = []
        for pdf_file in pdf_files:
            meta = self.extract_paper(pdf_file)
            results.append(meta)

        return results

    @staticmethod
    def save_to_json(papers: List[PaperMetadata], output_path: Union[str, Path]) -> None:
        """Serialize extracted papers to structured JSON matching the project schema."""
        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "papers": [paper.to_dict() for paper in papers]
        }

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        logger.info("Saved extracted papers JSON to %s (%d papers)", out_path, len(papers))

