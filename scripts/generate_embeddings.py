"""Script to generate dense vector embeddings for all chunks in data/chunked/chunks.json
and save embeddings.npy and embedding_metadata.json.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from embeddings import (
    Embedder,
    EmbeddingConfig,
    validate_embeddings,
)


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging for embedding generation."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate dense vector embeddings for research paper chunks using Sentence Transformers."
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default="data/chunked/chunks.json",
        help="Path to chunked dataset (default: data/chunked/chunks.json)",
    )
    parser.add_argument(
        "--output-npy",
        type=str,
        default="data/embeddings/embeddings.npy",
        help="Path to output numpy array (default: data/embeddings/embeddings.npy)",
    )
    parser.add_argument(
        "--output-metadata",
        type=str,
        default="data/embeddings/embedding_metadata.json",
        help="Path to output embedding metadata (default: data/embeddings/embedding_metadata.json)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Sentence Transformers embedding model (default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for inference (cpu, cuda, etc. Default: auto)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing embeddings and metadata files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("generate_embeddings")

    input_path = Path(args.input_file)
    output_npy = Path(args.output_npy)
    output_meta = Path(args.output_metadata)

    logger.info("=" * 70)
    logger.info("THE RELAY: Dense Embedding Generation Pipeline")
    logger.info("=" * 70)
    logger.info("Input file:      %s", input_path.resolve())
    logger.info("Output npy:      %s", output_npy.resolve())
    logger.info("Output metadata: %s", output_meta.resolve())
    logger.info("Model:           %s", args.model_name)
    logger.info("Batch size:      %d", args.batch_size)

    # Overwrite check
    if (output_npy.exists() or output_meta.exists()) and not args.overwrite:
        logger.error(
            "Output embeddings or metadata file already exists! Use --overwrite to overwrite."
        )
        return 1

    # Load input chunks
    if not input_path.exists():
        logger.error("Input chunks file does not exist: %s", input_path)
        return 1

    try:
        with open(input_path, "r", encoding="utf-8") as f:
            chunk_payload = json.load(f)
        chunks = chunk_payload.get("chunks", [])
    except Exception as exc:
        logger.error("Failed to load chunked dataset: %s", exc, exc_info=True)
        return 1

    if not chunks:
        logger.error("No chunks found in %s", input_path)
        return 1

    logger.info("Loaded %d chunks from %s", len(chunks), input_path)

    # Initialize Embedder
    config = EmbeddingConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        device=args.device,
    )
    embedder = Embedder(config)

    # Generate embeddings
    start_time = time.perf_counter()
    try:
        embeddings, metadata_list = embedder.embed_chunks(chunks)
    except Exception as exc:
        logger.error("Failed to generate embeddings: %s", exc, exc_info=True)
        return 1
    elapsed_time = time.perf_counter() - start_time

    logger.info(
        "Generated %d embeddings in %.2f seconds (%.1f chunks/sec)",
        embeddings.shape[0],
        elapsed_time,
        embeddings.shape[0] / elapsed_time if elapsed_time > 0 else 0,
    )

    # Validate embeddings
    validation_report = validate_embeddings(
        embeddings=embeddings,
        metadata_list=metadata_list,
        expected_count=len(chunks),
        expected_dim=config.expected_dimension,
    )

    # Save outputs
    try:
        embedder.save_embeddings(
            embeddings=embeddings,
            metadata=metadata_list,
            npy_path=output_npy,
            meta_path=output_meta,
            config=config,
        )
    except Exception as exc:
        logger.error("Failed to save embeddings or metadata: %s", exc, exc_info=True)
        return 1

    # Print summary
    print("\n" + "=" * 70)
    print("EMBEDDING GENERATION SUMMARY")
    print("=" * 70)
    print(f"Embedding Model:         {config.model_name}")
    print(f"Embedding Dimension:     {embeddings.shape[1]}")
    print(f"Total Chunks Embedded:   {embeddings.shape[0]}")
    print(f"Batch Size:              {config.batch_size}")
    print(f"Generation Time:         {elapsed_time:.2f} seconds")
    print(f"Throughput:              {embeddings.shape[0] / elapsed_time:.1f} chunks/sec")
    print(f"Embeddings Array Path:   {output_npy.resolve()}")
    print(f"Metadata File Path:      {output_meta.resolve()}")
    print("-" * 70)

    print("\nVALIDATION REPORT:")
    print(validation_report.summary())
    print("=" * 70 + "\n")

    if not validation_report.is_valid:
        logger.error("Embedding validation failed! Review the errors above.")
        return 1

    logger.info("Embedding generation and validation completed successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

