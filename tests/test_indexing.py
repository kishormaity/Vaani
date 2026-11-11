"""Unit tests for the embedding generation and FAISS vector indexing modules."""

import shutil
import tempfile
import unittest
from pathlib import Path

import numpy as np

from embeddings import (
    ChunkVectorMetadata,
    EmbeddingConfig,
    validate_embeddings,
)
from vector_store import FAISSVectorStore, SearchResult


class TestEmbeddingsAndVectorStore(unittest.TestCase):
    """Test suite covering embedding validation, FAISS indexing, persistence, and cosine search."""

    def setUp(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.dim = 384
        self.num_samples = 10

        # Generate synthetic normalized vectors for fast, offline unit testing
        rng = np.random.default_rng(seed=42)
        raw_vectors = rng.standard_normal((self.num_samples, self.dim)).astype(np.float32)
        norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
        self.mock_vectors = raw_vectors / norms

        self.mock_metadata = [
            {
                "vector_index": i,
                "chunk_id": f"paper_1_c{i+1:04d}",
                "paper_id": "1706.03762",
                "title": "Attention Is All You Need",
                "version": 7,
                "source_filename": "attention.pdf",
                "source_page_numbers": [1 if i < 5 else 2],
                "chunk_index": i + 1,
                "token_count": 450,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "embedding_dimension": self.dim,
                "text": f"Sample chunk text for index {i}.",
            }
            for i in range(self.num_samples)
        ]

    def tearDown(self) -> None:
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_empty_input(self) -> None:
        """Verify handling of empty inputs in vector store and validation."""
        store = FAISSVectorStore(dimension=self.dim)
        self.assertEqual(store.total_vectors, 0)
        results = store.search(np.zeros(self.dim, dtype=np.float32), top_k=5)
        self.assertEqual(results, [])

    def test_normalization_check(self) -> None:
        """Verify that L2 normalization validates correctly on unit vectors."""
        report = validate_embeddings(
            self.mock_vectors,
            self.mock_metadata,
            expected_count=self.num_samples,
            expected_dim=self.dim,
        )
        self.assertTrue(report.is_valid, report.summary())
        self.assertLess(report.details["max_norm_deviation"], 1e-4)

    def test_metadata_to_vector_alignment(self) -> None:
        """Verify that vector indices align exactly 1-to-1 with metadata records."""
        store = FAISSVectorStore(dimension=self.dim)
        store.add_embeddings(self.mock_vectors, self.mock_metadata)

        self.assertEqual(store.total_vectors, self.num_samples)
        for i in range(self.num_samples):
            self.assertEqual(store.metadata_catalog[i]["vector_index"], i)
            self.assertEqual(store.metadata_catalog[i]["chunk_id"], f"paper_1_c{i+1:04d}")

    def test_invalid_embedding_values_detected(self) -> None:
        """Verify validation catches NaNs, Infs, wrong dimensions, and unnormalized vectors."""
        # 1. NaN check
        bad_nan = self.mock_vectors.copy()
        bad_nan[0, 0] = np.nan
        rep_nan = validate_embeddings(bad_nan, self.mock_metadata, self.num_samples, self.dim)
        self.assertFalse(rep_nan.is_valid)

        # 2. Inf check
        bad_inf = self.mock_vectors.copy()
        bad_inf[0, 0] = np.inf
        rep_inf = validate_embeddings(bad_inf, self.mock_metadata, self.num_samples, self.dim)
        self.assertFalse(rep_inf.is_valid)

        # 3. Non-normalized check
        bad_norm = self.mock_vectors.copy()
        bad_norm[0] = bad_norm[0] * 5.0
        rep_norm = validate_embeddings(bad_norm, self.mock_metadata, self.num_samples, self.dim)
        self.assertFalse(rep_norm.is_valid)

        # 4. Wrong dimension check
        bad_dim = self.mock_vectors[:, :100]
        rep_dim = validate_embeddings(bad_dim, self.mock_metadata, self.num_samples, self.dim)
        self.assertFalse(rep_dim.is_valid)

    def test_faiss_insertion_and_search(self) -> None:
        """Verify adding vectors and retrieving top matching chunks with cosine similarity."""
        store = FAISSVectorStore(dimension=self.dim)
        store.add_embeddings(self.mock_vectors, self.mock_metadata)

        # Querying with vector 0 should retrieve vector 0 as the top match with score ~ 1.0
        query_v = self.mock_vectors[0]
        results = store.search(query_v, top_k=3)

        self.assertEqual(len(results), 3)
        self.assertEqual(results[0].vector_index, 0)
        self.assertEqual(results[0].chunk_id, "paper_1_c0001")
        self.assertAlmostEqual(results[0].similarity_score, 1.0, places=4)

    def test_index_persistence_and_reload(self) -> None:
        """Verify that index and catalog can be saved and reloaded in a fresh store instance."""
        store = FAISSVectorStore(dimension=self.dim)
        store.add_embeddings(self.mock_vectors, self.mock_metadata)

        index_file = self.temp_dir / "faiss.index"
        catalog_file = self.temp_dir / "catalog.json"

        store.save(index_path=index_file, catalog_path=catalog_file)
        self.assertTrue(index_file.exists())
        self.assertTrue(catalog_file.exists())

        # Load fresh store
        reloaded = FAISSVectorStore.load(index_path=index_file, catalog_path=catalog_file)
        self.assertEqual(reloaded.total_vectors, self.num_samples)
        self.assertEqual(len(reloaded.metadata_catalog), self.num_samples)

        # Search on reloaded store must match
        res_orig = store.search(self.mock_vectors[2], top_k=3)
        res_reloaded = reloaded.search(self.mock_vectors[2], top_k=3)

        self.assertEqual(res_orig[0].chunk_id, res_reloaded[0].chunk_id)
        self.assertAlmostEqual(res_orig[0].similarity_score, res_reloaded[0].similarity_score, places=4)


if __name__ == "__main__":
    unittest.main()

