"""Script to build a persistent FAISS index from dense chunk embeddings
and execute a vector search smoke test.
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

import numpy as np

from embeddings import Embedder, EmbeddingConfig
from vector_store import FAISSVectorStore


def setup_logging(verbose: bool = False) -> None:
    """Configure structured logging for index building."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and validate a persistent FAISS vector index from chunk embeddings."
    )
    parser.add_argument(
        "--embeddings-npy",
        type=str,
        default="data/embeddings/embeddings.npy",
        help="Path to embeddings numpy array (default: data/embeddings/embeddings.npy)",
    )
    parser.add_argument(
        "--embeddings-metadata",
        type=str,
        default="data/embeddings/embedding_metadata.json",
        help="Path to embedding metadata (default: data/embeddings/embedding_metadata.json)",
    )
    parser.add_argument(
        "--chunks-file",
        type=str,
        default="data/chunked/chunks.json",
        help="Path to original chunks JSON to link chunk text (default: data/chunked/chunks.json)",
    )
    parser.add_argument(
        "--output-index",
        type=str,
        default="data/vector_store/faiss.index",
        help="Path to save FAISS index (default: data/vector_store/faiss.index)",
    )
    parser.add_argument(
        "--output-catalog",
        type=str,
        default="data/vector_store/chunk_catalog.json",
        help="Path to save vector store catalog (default: data/vector_store/chunk_catalog.json)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model name (default: sentence-transformers/all-MiniLM-L6-v2)",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=True,
        help="Run index smoke test query (default: True)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What is the Transformer architecture?",
        help="Query for smoke test (default: 'What is the Transformer architecture?')",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of results for smoke test (default: 5)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing FAISS index and catalog",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("build_index")

    p_npy = Path(args.embeddings_npy)
    p_meta = Path(args.embeddings_metadata)
    p_chunks = Path(args.chunks_file)
    p_index = Path(args.output_index)
    p_catalog = Path(args.output_catalog)

    logger.info("=" * 70)
    logger.info("THE RELAY: FAISS Vector Index Builder")
    logger.info("=" * 70)
    logger.info("Embeddings:     %s", p_npy.resolve())
    logger.info("Metadata:       %s", p_meta.resolve())
    logger.info("Output Index:   %s", p_index.resolve())
    logger.info("Output Catalog: %s", p_catalog.resolve())

    # Check overwrite
    if (p_index.exists() or p_catalog.exists()) and not args.overwrite:
        logger.error(
            "Index or catalog file already exists! Use --overwrite to overwrite."
        )
        return 1

    # Load embeddings and metadata
    if not p_npy.exists() or not p_meta.exists():
        logger.error("Embeddings or metadata file not found! Run generate_embeddings.py first.")
        return 1

    try:
        embeddings = np.load(p_npy)
        with open(p_meta, "r", encoding="utf-8") as f:
            meta_json = json.load(f)
        metadata_list = meta_json.get("embeddings", meta_json if isinstance(meta_json, list) else [])
    except Exception as exc:
        logger.error("Failed to load embeddings or metadata: %s", exc, exc_info=True)
        return 1

    logger.info("Loaded embeddings matrix: shape %s", embeddings.shape)
    logger.info("Loaded %d metadata records", len(metadata_list))

    # Optional: Attach chunk text from chunks.json for rich smoke test output
    if p_chunks.exists():
        try:
            with open(p_chunks, "r", encoding="utf-8") as f:
                chunk_data = json.load(f)
            text_by_id = {c["chunk_id"]: c.get("text", "") for c in chunk_data.get("chunks", [])}
            for m in metadata_list:
                cid = m.get("chunk_id")
                if cid in text_by_id:
                    m["text"] = text_by_id[cid]
            logger.info("Enriched metadata with chunk texts from %s", p_chunks)
        except Exception as exc:
            logger.warning("Could not attach chunk text: %s", exc)

    # Initialize Vector Store
    dimension = int(embeddings.shape[1])
    store = FAISSVectorStore(dimension=dimension, model_name=args.model_name)

    # Add embeddings to FAISS
    try:
        store.add_embeddings(embeddings, metadata_list)
    except Exception as exc:
        logger.error("Failed to add embeddings to FAISS index: %s", exc, exc_info=True)
        return 1

    # Save index and catalog
    try:
        store.save(index_path=p_index, catalog_path=p_catalog)
    except Exception as exc:
        logger.error("Failed to save FAISS index: %s", exc, exc_info=True)
        return 1

    # Validation: Verify index can be reloaded in a fresh instance
    logger.info("Validating index reload from disk...")
    try:
        reloaded_store = FAISSVectorStore.load(index_path=p_index, catalog_path=p_catalog)
        if reloaded_store.total_vectors != store.total_vectors:
            logger.error(
                "Reloaded index count (%d) mismatch original (%d)",
                reloaded_store.total_vectors,
                store.total_vectors,
            )
            return 1
        logger.info("Index reload validation PASSED (%d vectors).", reloaded_store.total_vectors)
    except Exception as exc:
        logger.error("Index reload validation FAILED: %s", exc, exc_info=True)
        return 1

    # Index Summary
    index_size_kb = p_index.stat().st_size / 1024.0
    catalog_size_kb = p_catalog.stat().st_size / 1024.0

    print("\n" + "=" * 70)
    print("FAISS VECTOR INDEX SUMMARY")
    print("=" * 70)
    print(f"Index Type:              faiss.IndexFlatIP (Cosine Similarity)")
    print(f"Embedding Model:         {args.model_name}")
    print(f"Vector Dimension:        {dimension}")
    print(f"Total Indexed Vectors:   {store.total_vectors}")
    print(f"FAISS Index Path:        {p_index.resolve()}")
    print(f"FAISS Index Size:        {index_size_kb:.2f} KB")
    print(f"Catalog Path:            {p_catalog.resolve()}")
    print(f"Catalog Size:            {catalog_size_kb:.2f} KB")
    print("-" * 70)

    # Execute Smoke Test Query
    if args.smoke_test:
        print("\n" + "=" * 70)
        print("VECTOR SEARCH SMOKE TEST")
        print("=" * 70)
        print(f"Query: \"{args.query}\"")
        print(f"Top-K: {args.top_k}")
        print("-" * 70)

        embed_config = EmbeddingConfig(model_name=args.model_name, normalize_embeddings=True)
        embedder = Embedder(embed_config)
        query_vector = embedder.embed_texts([args.query])[0]

        results = reloaded_store.search(query_vector=query_vector, top_k=args.top_k)

        print(f"{'Rank':<4} | {'Score':<6} | {'Chunk ID':<18} | {'Pages':<7} | {'Paper Title'}")
        print("-" * 70)
        for rank, res in enumerate(results, start=1):
            pages_str = str(res.source_page_numbers)
            title_trunc = res.title[:35] + ("..." if len(res.title) > 35 else "")
            print(f"{rank:<4} | {res.similarity_score:.4f} | {res.chunk_id:<18} | {pages_str:<7} | {title_trunc}")

        print("-" * 70)
        if results:
            print("\nTop 1 Result Snippet Preview:")
            top1_text = results[0].text[:300].strip()
            print(f"[{results[0].chunk_id}] {top1_text}...\n")
        print("=" * 70 + "\n")

    logger.info("FAISS vector index built and verified successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

