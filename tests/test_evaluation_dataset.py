"""Unit tests for Phase 5 Benchmark Evaluation Dataset (data/evaluation_questions.json)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class TestEvaluationDataset(unittest.TestCase):
    """Validation test suite for data/evaluation_questions.json."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.eval_path = PROJECT_ROOT / "data" / "evaluation_questions.json"
        cls.papers_path = PROJECT_ROOT / "data" / "extracted" / "papers.json"

        cls.assertTrue(cls.eval_path.exists(), f"Evaluation questions file missing at {cls.eval_path}")
        cls.assertTrue(cls.papers_path.exists(), f"Extracted papers file missing at {cls.papers_path}")

        with open(cls.eval_path, "r", encoding="utf-8") as f:
            cls.questions = json.load(f)

        with open(cls.papers_path, "r", encoding="utf-8") as f:
            papers_data = json.load(f)["papers"]
            cls.corpus = {
                f"{p['paper_id']}v{p['version']}": {pg["page_number"] for pg in p["pages"]}
                for p in papers_data
            }

    def test_total_count(self) -> None:
        """Verify that the evaluation dataset contains exactly 15 questions."""
        self.assertEqual(len(self.questions), 15)

    def test_unique_identifiers(self) -> None:
        """Verify that all question IDs are unique and sequentially formatted (q01-q15)."""
        ids = [q["id"] for q in self.questions]
        self.assertEqual(len(ids), len(set(ids)))
        expected_ids = [f"q{i:02d}" for i in range(1, 16)]
        self.assertEqual(ids, expected_ids)

    def test_category_distribution(self) -> None:
        """Verify category breakdown across factual, comparative, and unanswerable."""
        categories = [q["category"] for q in self.questions]
        self.assertEqual(categories.count("factual"), 6)
        self.assertEqual(categories.count("comparative"), 5)
        self.assertEqual(categories.count("unanswerable"), 4)

    def test_schema_completeness(self) -> None:
        """Verify all records conform strictly to the required schema."""
        required_keys = {
            "id",
            "category",
            "target_papers",
            "question",
            "ground_truth_answer",
            "expected_citations",
            "should_refuse",
        }

        for q in self.questions:
            self.assertEqual(set(q.keys()), required_keys, f"Schema mismatch in question {q['id']}")
            self.assertIsInstance(q["id"], str)
            self.assertIn(q["category"], {"factual", "comparative", "unanswerable"})
            self.assertIsInstance(q["target_papers"], list)
            self.assertIsInstance(q["question"], str)
            self.assertTrue(len(q["question"].strip()) > 10)
            self.assertIsInstance(q["ground_truth_answer"], str)
            self.assertTrue(len(q["ground_truth_answer"].strip()) > 10)
            self.assertIsInstance(q["expected_citations"], list)
            self.assertIsInstance(q["should_refuse"], bool)

    def test_expected_citations_validity(self) -> None:
        """Verify that every cited arXiv ID and page number exists in the extracted paper corpus."""
        for q in self.questions:
            citations = q["expected_citations"]
            if q["should_refuse"]:
                self.assertEqual(
                    len(citations),
                    0,
                    f"Unanswerable question {q['id']} must have empty expected_citations",
                )
            else:
                self.assertGreater(
                    len(citations),
                    0,
                    f"Answerable question {q['id']} must have at least one expected citation",
                )
                for cite in citations:
                    arxiv_id = cite["arxiv_id"]
                    page = cite["page"]
                    self.assertIn(
                        arxiv_id,
                        self.corpus,
                        f"Question {q['id']} cites unknown paper version {arxiv_id}",
                    )
                    self.assertIn(
                        page,
                        self.corpus[arxiv_id],
                        f"Question {q['id']} cites page {page} not found in {arxiv_id}",
                    )

    def test_refusal_consistency(self) -> None:
        """Verify that should_refuse is True if and only if category is unanswerable."""
        for q in self.questions:
            if q["category"] == "unanswerable":
                self.assertTrue(q["should_refuse"], f"Question {q['id']} is unanswerable but should_refuse is False")
            else:
                self.assertFalse(q["should_refuse"], f"Question {q['id']} is {q['category']} but should_refuse is True")


if __name__ == "__main__":
    unittest.main()

