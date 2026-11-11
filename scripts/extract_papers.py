"""Script to extract text and metadata from research papers in data/papers/
and save structured JSON to data/extracted/papers.json.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data_extraction import (
    PDFExtractor,
    PaperValidationError,
    validate_extraction,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging for extraction script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Extract text and metadata from research paper PDFs using PyMuPDF."
    )
    parser.add_argument(
        "--papers-dir",
        type=str,
        default="data/papers",
        help="Directory containing the input research papers (default: data/papers)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/extracted/papers.json",
        help="Destination JSON file (default: data/extracted/papers.json)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("extract_papers")

    papers_dir = Path(args.papers_dir)
    output_path = Path(args.output_file)

    logger.info("=" * 70)
    logger.info("THE RELAY: Research Paper PDF Text Extraction")
    logger.info("=" * 70)
    logger.info("Input directory: %s", papers_dir.resolve())
    logger.info("Output JSON:     %s", output_path.resolve())

    # Step 1: Perform PDF Extraction
    extractor = PDFExtractor()
    try:
        extracted_papers = extractor.extract_directory(papers_dir)
    except Exception as exc:
        logger.error("Fatal extraction error: %s", exc, exc_info=True)
        return 1

    if not extracted_papers:
        logger.error("No papers were extracted from %s", papers_dir)
        return 1

    # Step 2: Save to JSON
    try:
        extractor.save_to_json(extracted_papers, output_path)
    except Exception as exc:
        logger.error("Failed to save JSON to %s: %s", output_path, exc, exc_info=True)
        return 1

    # Step 3: Run Validation Suite
    logger.info("=" * 70)
    logger.info("Running Validation Suite...")
    logger.info("=" * 70)
    validation_report = validate_extraction(extracted_papers, json_path=output_path)

    # Step 4: Final Summary & Reporting
    print("\n" + "=" * 70)
    print("EXTRACTION SUMMARY")
    print("=" * 70)
    print(f"{'Paper ID':<12} | {'Ver':<4} | {'Pages':<7} | {'Extracted':<9} | {'Empty':<6} | {'Status':<8} | {'Title'}")
    print("-" * 70)
    for p in extracted_papers:
        ver_str = f"v{p.version}" if p.version is not None else "N/A"
        pages_str = f"{p.extracted_pages}/{p.total_pages}"
        empty_str = str(len(p.empty_pages))
        title_trunc = p.title[:30] + ("..." if len(p.title) > 30 else "")
        print(f"{p.paper_id:<12} | {ver_str:<4} | {p.total_pages:<7} | {pages_str:<9} | {empty_str:<6} | {p.status:<8} | {title_trunc}")
    print("-" * 70)

    print("\nVALIDATION REPORT:")
    print(validation_report.summary())
    print("=" * 70 + "\n")

    if not validation_report.is_valid:
        logger.error("Validation failed! Please review the errors listed above.")
        return 1

    logger.info("All checks passed successfully. Extraction output is ready for RAG pipeline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

