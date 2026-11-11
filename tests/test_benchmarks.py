"""Unit tests for Benchmark Evaluator metrics (benchmarks/evaluator.py)."""

from __future__ import annotations

import unittest

from benchmarks.evaluator import (
    compute_citation_metrics,
    evaluate_refusal,
    score_answer_accuracy,
)


class TestBenchmarkEvaluator(unittest.TestCase):
    """Test suite for citation metrics, refusal evaluation, and answer scoring."""

    def test_citation_metrics_exact_match(self) -> None:
        """Verify exact citation matching on arXiv ID and page numbers."""
        expected = [{"arxiv_id": "1706.03762v7", "page": 4}]
        predicted = ["[arXiv:1706.03762v7, p.4]"]

        prec, rec, f1 = compute_citation_metrics(predicted, expected)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)
        self.assertEqual(f1, 1.0)

    def test_citation_metrics_version_normalization(self) -> None:
        """Verify that paper IDs match regardless of version suffix (e.g. v7 vs unversioned)."""
        expected = [{"arxiv_id": "1706.03762", "page": 9}]
        predicted = ["[arXiv:1706.03762v7, p.9]"]

        prec, rec, f1 = compute_citation_metrics(predicted, expected)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)
        self.assertEqual(f1, 1.0)

    def test_citation_metrics_multiple_and_partial(self) -> None:
        """Verify multi-citation and partial precision/recall calculation."""
        expected = [
            {"arxiv_id": "1706.03762", "page": 4},
            {"arxiv_id": "1706.03762", "page": 5},
        ]
        # Predicted only cites page 4
        predicted = ["[arXiv:1706.03762, p.4]"]
        prec, rec, f1 = compute_citation_metrics(predicted, expected)
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 0.5)
        self.assertAlmostEqual(f1, 0.6667, places=4)

        # Predicted cites page 4 and page 10 (page 10 not in expected)
        predicted_extra = ["[arXiv:1706.03762, p.4]", "[arXiv:1706.03762, p.10]"]
        prec2, rec2, f12 = compute_citation_metrics(predicted_extra, expected)
        self.assertEqual(prec2, 0.5)
        self.assertEqual(rec2, 0.5)
        self.assertEqual(f12, 0.5)

    def test_citation_metrics_refusal_edge_cases(self) -> None:
        """Verify edge case handling for unanswerable / refusal questions."""
        # Both empty: legitimate refusal
        prec, rec, f1 = compute_citation_metrics([], [])
        self.assertEqual(prec, 1.0)
        self.assertEqual(rec, 1.0)
        self.assertEqual(f1, 1.0)

        # Expected citations present, but model generated none
        prec_miss, rec_miss, f1_miss = compute_citation_metrics([], [{"arxiv_id": "1706.03762", "page": 4}])
        self.assertEqual(prec_miss, 0.0)
        self.assertEqual(rec_miss, 0.0)
        self.assertEqual(f1_miss, 0.0)

        # Expected no citations, but model hallucinated one
        prec_hallu, rec_hallu, f1_hallu = compute_citation_metrics(["[arXiv:1706.03762, p.4]"], [])
        self.assertEqual(prec_hallu, 0.0)
        self.assertEqual(rec_hallu, 1.0)
        self.assertEqual(f1_hallu, 0.0)

    def test_evaluate_refusal(self) -> None:
        """Verify evaluate_refusal boolean logic."""
        self.assertTrue(evaluate_refusal(True, True))
        self.assertTrue(evaluate_refusal(False, False))
        self.assertFalse(evaluate_refusal(True, False))
        self.assertFalse(evaluate_refusal(False, True))

    def test_score_answer_accuracy_exact_and_semantic(self) -> None:
        """Verify answer accuracy scoring on factual overlaps."""
        gt = (
            "Scaled Dot-Product Attention is computed as Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V. "
            "The scaling factor 1/sqrt(d_k) is applied because for large values of d_k, the dot products grow large."
        )
        # Exact match
        score_exact = score_answer_accuracy(gt, gt)
        self.assertEqual(score_exact, 1.0)

        # Partial grounded match
        partial_pred = (
            "Scaled Dot-Product Attention computes Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V "
            "with scaling factor to prevent small gradients [arXiv:1706.03762v7, p.4]."
        )
        score_partial = score_answer_accuracy(partial_pred, gt)
        self.assertGreaterEqual(score_partial, 0.6)

        # Completely disjoint content
        disjoint_pred = "Chocolate chip cookies require flour, sugar, butter, and baking soda."
        score_disjoint = score_answer_accuracy(disjoint_pred, gt)
        self.assertEqual(score_disjoint, 0.0)

    def test_score_answer_accuracy_refusal(self) -> None:
        """Verify answer accuracy correctly scores refusal alignment."""
        refusal_gt = "The provided passages do not contain sufficient evidence to answer this question."
        refusal_pred = "I cannot answer this question based on the provided research papers."
        non_refusal_pred = "Quantum annealing is a metaheuristic algorithm for optimization."

        # Both refusal -> 1.0
        self.assertEqual(score_answer_accuracy(refusal_pred, refusal_gt), 1.0)
        # GT refusal, pred attempted answer -> 0.0
        self.assertEqual(score_answer_accuracy(non_refusal_pred, refusal_gt), 0.0)


if __name__ == "__main__":
    unittest.main()
