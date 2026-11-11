"""Script to chunk extracted research papers into configurable, metadata-preserving chunks
and save structured JSON to data/chunked/chunks.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from chunking import (
    ChunkConfig,
    ParagraphAwareChunker,
    validate_chunks,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging for chunking script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk research paper text into configurable token segments preserving citation metadata."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/extracted/papers.json",
        help="Path to extracted papers JSON (default: data/extracted/papers.json)",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="data/chunked/chunks.json",
        help="Path to save chunked dataset (default: data/chunked/chunks.json)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500,
        help="Target chunk size in tokens (default: 500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=75,
        help="Token overlap between consecutive chunks (default: 75)",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="cl100k_base",
        help="Tiktoken tokenizer encoding (default: cl100k_base)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow overwriting an existing output file",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("chunk_papers")

    input_path = Path(args.input_file)
    output_path = Path(args.output_file)

    logger.info("=" * 70)
    logger.info("THE RELAY: Research Paper Chunking Pipeline")
    logger.info("=" * 70)
    logger.info("Input file:       %s", input_path.resolve())
    logger.info("Output file:      %s", output_path.resolve())
    logger.info("Chunk size:       %d tokens", args.chunk_size)
    logger.info("Chunk overlap:    %d tokens", args.chunk_overlap)
    logger.info("Tokenizer:        %s", args.tokenizer)

    # Safety: Do not overwrite an existing experiment's output without explicit intention
    if output_path.exists() and not args.overwrite:
        logger.error(
            "Output file '%s' already exists! To prevent accidental overwriting of experiment "
            "data, use the --overwrite flag or specify a different --output-file.",
            output_path,
        )
        return 1

    # Load input papers
    if not input_path.exists():
        logger.error("Input file does not exist: %s", input_path)
        return 1

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            papers_payload = json.load(f)
    except Exception as exc:
        logger.error("Failed to load input JSON '%s': %s", input_path, exc, exc_info=True)
        return 1

    # Initialize configuration & chunker
    try:
        config = ChunkConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            tokenizer_name=args.tokenizer,
        )
    except Exception as conf_err:
        logger.error("Invalid chunking configuration: %s", conf_err)
        return 1

    chunker = ParagraphAwareChunker(config)

    # Execute chunking
    try:
        all_chunks, stats = chunker.chunk_dataset(papers_payload)
    except Exception as exc:
        logger.error("Chunking failed: %s", exc, exc_info=True)
        return 1

    # Assemble output payload
    payload = {
        "configuration": config.to_dict(),
        "statistics": stats.to_dict(),
        "chunks": [chunk.to_dict() for chunk in all_chunks],
    }

    # Save to disk
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info("Saved %d chunks to %s", len(all_chunks), output_path)
    except Exception as exc:
        logger.error("Failed to write output JSON to '%s': %s", output_path, exc, exc_info=True)
        return 1

    # Execute validation suite
    logger.info("=" * 70)
    logger.info("Running Chunk Validation Suite...")
    logger.info("=" * 70)
    validation_report = validate_chunks(payload, source_papers_payload=papers_payload)

    # Print summary
    print("\n" + "=" * 70)
    print("CHUNKING SUMMARY & STATISTICS")
    print("=" * 70)
    print(f"Total Papers Processed:    {stats.total_papers}")
    print(f"Total Source Pages:        {stats.total_source_pages}")
    print(f"Total Chunks Generated:    {stats.total_chunks}")
    print(f"Average Tokens / Chunk:    {stats.average_tokens_per_chunk}")
    print(f"Min Tokens in a Chunk:     {stats.min_tokens_per_chunk}")
    print(f"Max Tokens in a Chunk:     {stats.max_tokens_per_chunk}")
    print(f"Chunks Exceeding Limit:    {stats.chunks_exceeding_limit}")
    print("-" * 70)
    print("Chunks Generated Per Paper:")
    for pid, count in stats.chunks_per_paper.items():
        print(f"  - {pid}: {count} chunks")
    print("-" * 70)

    print("\nVALIDATION REPORT:")
    print(validation_report.summary())
    print("=" * 70 + "\n")

    if not validation_report.is_valid:
        logger.error("Chunk validation failed! Please inspect the issues listed above.")
        return 1

    logger.info("All chunk validation checks passed. Chunked dataset is ready for embedding.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

