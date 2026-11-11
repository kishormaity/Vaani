"""Comprehensive unit and integration tests for the 4-step RAG pipeline orchestrator."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock

from llm.base import CompletionResult, RateLimitTelemetry
from llm.standin_service import StandinConfig, StandinLLMService
from pipeline.answer_generator import AnswerGenerator, GenerationResult
from pipeline.citation_verifier import CitationVerifier, VerificationResult
from pipeline.classifier import ClassificationResult, QueryClassifier
from pipeline.orchestrator import (
    SAFE_OUT_OF_SCOPE_REFUSAL,
    SAFE_VALIDATION_FAILURE_REFUSAL,
    PipelineResult,
    RAGOrchestrator,
)
from retrieval.retriever import RetrievedChunk, Retriever


def make_mock_chunk(
    chunk_id: str = "chunk_1",
    paper_id: str = "1706.03762",
    paper_title: str = "Attention Is All You Need",
    version: int = 7,
    page_numbers: list[int] = None,
    score: float = 0.85,
    text: str = "Multi-head attention mechanism details.",
) -> RetrievedChunk:
    pages = page_numbers or [4, 5]
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        paper_title=paper_title,
        version=version,
        page_numbers=pages,
        text=text,
        score=score,
        citation=f"[arXiv:{paper_id}v{version}, pp.{', '.join(str(p) for p in pages)}]",
    )


class TestPipelineComponents(unittest.TestCase):
    """Unit tests for Classifier and CitationVerifier."""

    def setUp(self) -> None:
        self.classifier = QueryClassifier()
        self.verifier = CitationVerifier()

    def test_classifier_intent_and_scope(self) -> None:
        """Verify query classifier detects types and filters out-of-scope queries."""
        # In-scope factual
        res1 = self.classifier.classify("What is the formula for scaled dot-product attention?")
        self.assertTrue(res1.is_in_scope)
        self.assertEqual(res1.question_type, "factual")
        self.assertIn("1706.03762", res1.target_papers)

        # In-scope comparison
        res2 = self.classifier.classify("Compare Moshi speech dialogue with standard text-to-speech models.")
        self.assertTrue(res2.is_in_scope)
        self.assertEqual(res2.question_type, "comparison")
        self.assertIn("2410.00037", res2.target_papers)

        # In-scope explanation
        res3 = self.classifier.classify("How does dynamic chunking determine token boundaries?")
        self.assertTrue(res3.is_in_scope)
        self.assertEqual(res3.question_type, "explanation")
        self.assertIn("2507.07955", res3.target_papers)

        # Clearly out-of-scope
        res4 = self.classifier.classify("How do I bake sourdough bread with active dry yeast?")
        self.assertFalse(res4.is_in_scope)

        res5 = self.classifier.classify("Explain quantum entanglement and quantum teleportation.")
        self.assertFalse(res5.is_in_scope)

    def test_citation_verifier_rules(self) -> None:
        """Verify citation verification handles valid, invalid paper, invalid page, and refusals."""
        chunks = [make_mock_chunk(paper_id="1706.03762", page_numbers=[4, 5])]

        # Valid citation
        good_text = "The Transformer relies on self-attention [arXiv:1706.03762v7, p.4]."
        res_good = self.verifier.verify(good_text, chunks)
        self.assertTrue(res_good.is_valid)
        self.assertFalse(res_good.is_refusal)
        self.assertEqual(len(res_good.valid_citations), 1)

        # Invalid page citation (cites page 12, but only [4, 5] in context)
        bad_page_text = "The Transformer relies on self-attention [arXiv:1706.03762v7, p.12]."
        res_bad_page = self.verifier.verify(bad_page_text, chunks)
        self.assertFalse(res_bad_page.is_valid)
        self.assertIn("12", res_bad_page.error_message)

        # Invalid paper ID
        bad_paper_text = "Quantum computing is fast [arXiv:9999.12345v1, p.1]."
        res_bad_paper = self.verifier.verify(bad_paper_text, chunks)
        self.assertFalse(res_bad_paper.is_valid)
        self.assertIn("9999.12345", res_bad_paper.error_message)

        # Legitimate refusal
        refusal_text = "The provided passages do not contain information regarding this topic."
        res_refusal = self.verifier.verify(refusal_text, chunks)
        self.assertTrue(res_refusal.is_valid)
        self.assertTrue(res_refusal.is_refusal)


class TestRAGOrchestrator(unittest.IsolatedAsyncioTestCase):
    """State machine tests for the 4-step RAG orchestrator with verification and retry loop."""

    async def test_normal_question_answering_with_valid_citations(self) -> None:
        """Test normal question answering: in-scope -> retrieve -> draft -> verify -> accepted."""
        mock_chunks = [make_mock_chunk(chunk_id="c1", paper_id="1706.03762", page_numbers=[4, 5], score=0.88)]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_chunks

        mock_llm = MagicMock()
        mock_llm.source = "mock-real"
        mock_completion = CompletionResult(
            content="Multi-head attention allows the model to attend to information [arXiv:1706.03762v7, p.4].",
            start_time=100.0,
            first_token_ms=45.0,
            total_ms=120.0,
            tokens_in=350,
            tokens_out=25,
            telemetry=RateLimitTelemetry(),
        )
        mock_llm.complete_stream = AsyncMock(return_value=mock_completion)

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_client=mock_llm,
        )

        result: PipelineResult = await orchestrator.execute("What is multi-head attention?")

        self.assertFalse(result.is_refusal)
        self.assertEqual(result.loop_count, 0)
        self.assertEqual(
            result.steps_executed,
            ["step_1_analyze", "step_2_retrieve", "step_3_draft", "step_4_verify", "accepted"],
        )
        self.assertIn("[arXiv:1706.03762v7, p.4]", result.answer)
        self.assertTrue(result.verification_state.is_valid)
        self.assertEqual(mock_llm.complete_stream.call_count, 1)
        self.assertEqual(result.telemetry["source"], "mock-real")
        self.assertGreater(result.telemetry["first_token_ms"], 0.0)

    async def test_out_of_scope_query_refusal(self) -> None:
        """Test out-of-scope query early refusal: skips retrieval and LLM generation."""
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_client=mock_llm,
        )

        result: PipelineResult = await orchestrator.execute("How do I bake chocolate chip cookies?")

        self.assertTrue(result.is_refusal)
        self.assertEqual(result.steps_executed, ["step_1_analyze", "step_1_refusal"])
        self.assertEqual(result.answer, SAFE_OUT_OF_SCOPE_REFUSAL)
        self.assertEqual(result.loop_count, 0)
        self.assertEqual(mock_retriever.retrieve.call_count, 0)
        self.assertEqual(mock_llm.complete_stream.call_count, 0)

    async def test_verification_failure_triggers_one_retry_loop(self) -> None:
        """Test verification failure triggering 1 retry loop with corrective feedback."""
        mock_chunks = [make_mock_chunk(chunk_id="c1", paper_id="1706.03762", page_numbers=[4, 5], score=0.85)]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_chunks

        # Attempt 0 returns bad page citation [p.99]
        bad_completion = CompletionResult(
            content="Multi-head attention mechanism [arXiv:1706.03762v7, p.99].",
            start_time=100.0,
            first_token_ms=50.0,
            total_ms=130.0,
            tokens_in=300,
            tokens_out=20,
            telemetry=RateLimitTelemetry(),
        )
        # Attempt 1 returns corrected page citation [p.4]
        corrected_completion = CompletionResult(
            content="Multi-head attention mechanism [arXiv:1706.03762v7, p.4].",
            start_time=101.0,
            first_token_ms=48.0,
            total_ms=125.0,
            tokens_in=350,
            tokens_out=20,
            telemetry=RateLimitTelemetry(),
        )

        mock_llm = MagicMock()
        mock_llm.source = "mock-groq"
        mock_llm.complete_stream = AsyncMock(side_effect=[bad_completion, corrected_completion])

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_client=mock_llm,
        )

        result: PipelineResult = await orchestrator.execute("Explain multi-head attention")

        self.assertFalse(result.is_refusal)
        self.assertEqual(result.loop_count, 1)
        self.assertEqual(
            result.steps_executed,
            [
                "step_1_analyze",
                "step_2_retrieve",
                "step_3_draft",
                "step_4_verify",
                "step_3_draft_retry",
                "step_4_verify_retry",
                "accepted_on_retry",
            ],
        )
        self.assertEqual(mock_llm.complete_stream.call_count, 2)
        self.assertIn("[arXiv:1706.03762v7, p.4]", result.answer)
        self.assertTrue(result.verification_state.is_valid)

    async def test_double_verification_failure_branches_to_safe_refusal(self) -> None:
        """Test double failure: invalid citations on both attempts branches to safe refusal."""
        mock_chunks = [make_mock_chunk(chunk_id="c1", paper_id="1706.03762", page_numbers=[4, 5], score=0.85)]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_chunks

        bad_completion_1 = CompletionResult(
            content="Bad draft without citations.",
            start_time=100.0,
            first_token_ms=40.0,
            total_ms=100.0,
            tokens_in=300,
            tokens_out=15,
            telemetry=RateLimitTelemetry(),
        )
        bad_completion_2 = CompletionResult(
            content="Still invalid citation [arXiv:9999.00000, p.1].",
            start_time=101.0,
            first_token_ms=40.0,
            total_ms=100.0,
            tokens_in=350,
            tokens_out=18,
            telemetry=RateLimitTelemetry(),
        )

        mock_llm = MagicMock()
        mock_llm.source = "mock-groq"
        mock_llm.complete_stream = AsyncMock(side_effect=[bad_completion_1, bad_completion_2])

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_client=mock_llm,
        )

        result: PipelineResult = await orchestrator.execute("Explain attention mechanism")

        self.assertTrue(result.is_refusal)
        self.assertEqual(result.loop_count, 1)
        self.assertEqual(result.answer, SAFE_VALIDATION_FAILURE_REFUSAL)
        self.assertIn("branch_safe_refusal", result.steps_executed)

    async def test_end_to_end_with_standin_service(self) -> None:
        """Verify RAGOrchestrator running end-to-end with StandinLLMService."""
        mock_chunks = [
            make_mock_chunk(
                chunk_id="c1",
                paper_id="1706.03762",
                page_numbers=[4, 5],
                score=0.88,
                text="The multi-head attention mechanism computes representations in parallel.",
            )
        ]

        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_chunks

        standin_service = StandinLLMService(
            config=StandinConfig(
                base_ttft_ms=10.0,
                ms_per_token=1.0,
                failure_rate=0.0,
                simulate_latency=True,
            )
        )

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_client=standin_service,
        )

        result: PipelineResult = await orchestrator.execute("What is the transformer multi-head attention?")

        self.assertFalse(result.is_refusal)
        self.assertEqual(result.loop_count, 0)
        self.assertEqual(result.telemetry["source"], "stand-in")
        self.assertIn("[arXiv:1706.03762", result.answer)
        self.assertGreater(result.telemetry["first_token_ms"], 0.0)
        self.assertGreater(result.telemetry["end_to_end_ms"], 0.0)


if __name__ == "__main__":
    unittest.main()
