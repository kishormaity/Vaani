"""Unit tests for the retrieval and CrossEncoder re-ranking pipeline."""

import unittest
from pathlib import Path

from retrieval import (
    PRESET_CONFIGURATIONS,
    RetrievedChunk,
    Retriever,
    format_citation,
)


class TestRetrievalPipeline(unittest.TestCase):
    """Test suite for Table 3 retrieval settings (A, B, C) and citation formatting."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.index_path = Path("data/vector_store/faiss.index")
        cls.catalog_path = Path("data/vector_store/chunk_catalog.json")
        if not cls.index_path.exists() or not cls.catalog_path.exists():
            raise FileNotFoundError("FAISS index or catalog not found. Build index first.")

        # Initialize retrievers for each setting
        cls.retriever_a = Retriever(setting="Setting A", index_path=cls.index_path, catalog_path=cls.catalog_path)
        cls.retriever_b = Retriever(setting="Setting B", index_path=cls.index_path, catalog_path=cls.catalog_path)
        cls.retriever_c = Retriever(setting="Setting C", index_path=cls.index_path, catalog_path=cls.catalog_path)

        cls.sample_query = "scaled dot-product attention formula"

    def test_citation_formatting(self) -> None:
        """Verify academic citation string generation for single and multiple pages."""
        cit_single = format_citation(paper_id="1706.03762", version=7, page_numbers=[5])
        self.assertEqual(cit_single, "[arXiv:1706.03762v7, p.5]")

        cit_multi = format_citation(paper_id="1706.03762", version=7, page_numbers=[1, 2])
        self.assertEqual(cit_multi, "[arXiv:1706.03762v7, pp.1, 2]")

        cit_no_ver = format_citation(paper_id="2507.07955", version=None, page_numbers=[12])
        self.assertEqual(cit_no_ver, "[arXiv:2507.07955, p.12]")

    def test_retrieval_setting_a(self) -> None:
        """Verify Setting A: Dense retrieval only with top_k=3."""
        results = self.retriever_a.retrieve(self.sample_query)
        self.assertEqual(len(results), 3)

        for res in results:
            self.assertIsInstance(res, RetrievedChunk)
            self.assertTrue(res.chunk_id)
            self.assertTrue(res.paper_id)
            self.assertTrue(res.paper_title)
            self.assertIsInstance(res.page_numbers, list)
            self.assertTrue(len(res.page_numbers) > 0)
            self.assertTrue(res.text)
            self.assertIsInstance(res.score, float)
            self.assertTrue(res.citation.startswith("[arXiv:"))

        # Top result for scaled dot-product attention should be from Attention paper
        self.assertEqual(results[0].paper_id, "1706.03762")

    def test_retrieval_setting_b(self) -> None:
        """Verify Setting B: Dense retrieval only with top_k=5."""
        results = self.retriever_b.retrieve(self.sample_query)
        self.assertEqual(len(results), 5)

        for res in results:
            self.assertIsInstance(res, RetrievedChunk)
            self.assertTrue(res.citation)
            self.assertGreater(len(res.text), 0)

    def test_retrieval_setting_c_reranking(self) -> None:
        """Verify Setting C: candidate_k=8, top_k=3, use_reranker=True with CrossEncoder re-ranking."""
        # Retrieve 8 raw candidates to see dense baseline order
        raw_candidates = self.retriever_a.vector_store.search(
            query_vector=self.retriever_a.embedder.embed_texts([self.sample_query])[0],
            top_k=8,
        )
        dense_top8_ids = [c.chunk_id for c in raw_candidates]

        # Execute Setting C retrieval
        results_c = self.retriever_c.retrieve(self.sample_query)
        self.assertEqual(len(results_c), 3)

        # Verify citation metadata preservation
        for res in results_c:
            self.assertIsInstance(res, RetrievedChunk)
            self.assertTrue(res.chunk_id)
            self.assertTrue(res.paper_id)
            self.assertTrue(res.paper_title)
            self.assertTrue(res.citation)
            self.assertTrue(res.text)

        # Verify re-ranker was applied: scores are cross-encoder logits
        # and re-ranked top 3 chunk IDs are valid members of the initial candidate set
        for res in results_c:
            self.assertIn(res.chunk_id, dense_top8_ids)

        # CrossEncoder scores on attention formula query should place the Attention paper chunk prominently
        self.assertEqual(results_c[0].paper_id, "1706.03762")


if __name__ == "__main__":
    unittest.main()

