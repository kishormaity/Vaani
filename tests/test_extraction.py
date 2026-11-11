"""Unit tests for the PDF extraction and metadata module."""

import unittest
from pathlib import Path
import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning, module="fitz")
    import fitz

from data_extraction import (
    PDFExtractor,
    VersionMismatchError,
    clean_text,
    validate_extraction,
)


class TestPDFExtraction(unittest.TestCase):
    """Test suite for PDF extraction, page numbering, and metadata verification."""

    def test_clean_text_whitespace_and_paragraphs(self):
        """Verify clean_text trims extra whitespace while preserving paragraph structure."""
        raw = "Line 1 with trailing spaces   \r\n\n\n\nLine 2 after paragraph break.\xa0With nbsp."
        cleaned = clean_text(raw)
        self.assertEqual(
            cleaned,
            "Line 1 with trailing spaces\n\nLine 2 after paragraph break. With nbsp.",
        )

    def test_clean_text_empty(self):
        """Verify clean_text correctly returns empty string for blank/whitespace content."""
        self.assertEqual(clean_text("   \n\n\t  \r\n "), "")

    def test_empty_page_preserves_original_page_numbers(self):
        """Verify that skipping an empty page preserves the original PDF page numbers.
        
        Example: If page 2 is empty, page 3 must retain page_number = 3 (not 2).
        """
        doc = fitz.open()
        p1 = doc.new_page()
        p1.insert_text((50, 50), "arXiv:1706.03762v7 Attention Is All You Need Page 1")

        # Page 2: empty/blank
        doc.new_page()

        # Page 3: text content
        p3 = doc.new_page()
        p3.insert_text((50, 50), "Page 3 Content")

        temp_pdf = Path("data/temp_test_empty_page.pdf")
        doc.save(str(temp_pdf))
        doc.close()

        try:
            extractor = PDFExtractor()
            result = extractor.extract_paper(temp_pdf)

            self.assertEqual(result.total_pages, 3)
            self.assertEqual(result.extracted_pages, 2)
            self.assertEqual(result.empty_pages, [2])

            # Assert original page numbers are preserved
            self.assertEqual(result.pages[0].page_number, 1)
            self.assertEqual(
                result.pages[1].page_number,
                3,
                "Page 3 must retain page_number=3 even when page 2 is empty!",
            )
        finally:
            if temp_pdf.exists():
                temp_pdf.unlink()

    def test_version_mismatch_detection(self):
        """Verify that a mismatch in paper version is detected and flagged."""
        doc = fitz.open()
        p1 = doc.new_page()
        # Simulated version 99 instead of expected version 7
        p1.insert_text((50, 50), "arXiv:1706.03762v99 Attention Is All You Need")

        temp_pdf = Path("data/temp_test_version_mismatch.pdf")
        doc.save(str(temp_pdf))
        doc.close()

        try:
            extractor = PDFExtractor()
            result = extractor.extract_paper(temp_pdf)
            self.assertIn("Version mismatch", "".join(result.errors))
        finally:
            if temp_pdf.exists():
                temp_pdf.unlink()

    def test_actual_papers_extraction_and_validation(self):
        """Verify that the actual 3 research papers pass all validation criteria."""
        extractor = PDFExtractor()
        papers = extractor.extract_directory("data/papers")
        self.assertEqual(len(papers), 3)

        report = validate_extraction(papers, json_path="data/extracted/papers.json")
        self.assertTrue(report.is_valid, report.summary())


if __name__ == "__main__":
    unittest.main()

