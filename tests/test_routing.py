"""Unit tests for query routing architecture across DIRECT_LLM, EXTERNAL_WEB, and CORPUS_RAG."""

from __future__ import annotations

import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from llm.base import CompletionResult, RateLimitTelemetry
from pipeline.citation_verifier import CitationVerifier, VerificationResult
from pipeline.classifier import ClassificationResult, QueryClassifier, QueryIntent
from pipeline.orchestrator import (
    SAFE_OUT_OF_SCOPE_REFUSAL,
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
    score: float = 0.88,
    text: str = "Scaled dot-product attention mechanism details.",
) -> RetrievedChunk:
    pages = page_numbers or [4]
    return RetrievedChunk(
        chunk_id=chunk_id,
        paper_id=paper_id,
        paper_title=paper_title,
        version=version,
        page_numbers=pages,
        text=text,
        score=score,
        citation=f"[arXiv:{paper_id}v{version}, p.4]",
    )


class TestQueryClassifierRouting(unittest.TestCase):
    """Test suite for QueryClassifier multi-route dispatch."""

    def setUp(self) -> None:
        self.classifier = QueryClassifier()

    def test_direct_llm_chitchat_routing(self) -> None:
        """Verify chit-chat and greetings route to DIRECT_LLM."""
        queries = [
            "Hello, who are you?",
            "Hi there!",
            "Good morning!",
            "What can you do?",
            "Tell me a joke",
            "Thanks for your help!",
        ]
        for q in queries:
            res = self.classifier.classify(q)
            self.assertEqual(
                res.intent,
                QueryIntent.DIRECT_LLM,
                f"Expected DIRECT_LLM for '{q}', got {res.intent}",
            )
            self.assertTrue(res.is_in_scope)

    def test_direct_llm_programming_routing(self) -> None:
        """Verify general programming help routes to DIRECT_LLM."""
        queries = [
            "Write a python function to reverse a string",
            "How do I sort a list of numbers in Python?",
            "Implement binary search algorithm in python",
            "Debug this syntax error in my bash script",
        ]
        for q in queries:
            res = self.classifier.classify(q)
            self.assertEqual(
                res.intent,
                QueryIntent.DIRECT_LLM,
                f"Expected DIRECT_LLM for '{q}', got {res.intent}",
            )
            self.assertTrue(res.is_in_scope)

    def test_external_web_routing(self) -> None:
        """Verify queries with URLs or explicit web fetch requests route to EXTERNAL_WEB."""
        res1 = self.classifier.classify("Fetch https://example.com and summarize it")
        self.assertEqual(res1.intent, QueryIntent.EXTERNAL_WEB)
        self.assertEqual(res1.external_url, "https://example.com")

        res2 = self.classifier.classify("Please browse to www.wikipedia.org and tell me the main topics")
        self.assertEqual(res2.intent, QueryIntent.EXTERNAL_WEB)
        self.assertEqual(res2.external_url, "https://www.wikipedia.org")

        res3 = self.classifier.classify("Search the web for the latest Python 3.14 features")
        self.assertEqual(res3.intent, QueryIntent.EXTERNAL_WEB)

    def test_corpus_rag_in_scope_routing(self) -> None:
        """Verify questions on verified research papers route to CORPUS_RAG."""
        queries = [
            ("Explain scaled dot-product attention", "1706.03762"),
            ("What is the conversational latency of Moshi?", "2410.00037"),
            ("How does dynamic chunking identify token segment boundaries?", "2507.07955"),
            ("Compare the Transformer self-attention with Moshi multi-stream dialogue", "1706.03762"),
        ]
        for q, expected_paper in queries:
            res = self.classifier.classify(q)
            self.assertEqual(res.intent, QueryIntent.CORPUS_RAG)
            self.assertTrue(res.is_in_scope)
            self.assertIn(expected_paper, res.target_papers)

    def test_corpus_rag_out_of_scope_refusal(self) -> None:
        """Verify unrelated non-corpus queries trigger safe refusal scope flag."""
        queries = [
            "How do I bake chocolate chip cookies?",
            "Explain quantum entanglement and quantum teleportation.",
            "What is the best sourdough bread recipe?",
        ]
        for q in queries:
            res = self.classifier.classify(q)
            self.assertEqual(res.intent, QueryIntent.CORPUS_RAG)
            self.assertFalse(res.is_in_scope)


class TestOrchestratorRouting(unittest.IsolatedAsyncioTestCase):
    """Test suite for RAGOrchestrator execution paths across the 3 routes."""

    async def test_direct_llm_execution_bypasses_retrieval_and_verifier(self) -> None:
        """Verify DIRECT_LLM route calls gateway directly without FAISS or CitationVerifier."""
        mock_retriever = MagicMock()
        mock_verifier = MagicMock()
        mock_gateway = MagicMock()

        direct_completion = CompletionResult(
            content="Hello! I am The Relay, an AI research assistant.",
            start_time=100.0,
            first_token_ms=45.0,
            total_ms=120.0,
            tokens_in=35,
            tokens_out=15,
            telemetry=RateLimitTelemetry(),
        )
        mock_gateway.complete_stream = AsyncMock(return_value=direct_completion)

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            citation_verifier=mock_verifier,
            gateway=mock_gateway,
        )

        result: PipelineResult = await orchestrator.run_pipeline("Hello, who are you?")

        # Assertions
        self.assertEqual(result.steps_executed, ["step_1_analyze", "direct_llm"])
        self.assertEqual(result.answer, "Hello! I am The Relay, an AI research assistant.")
        self.assertEqual(len(result.chunks), 0)
        self.assertFalse(result.is_refusal)
        self.assertEqual(result.classification.intent, QueryIntent.DIRECT_LLM)

        # Ensure retriever and citation verifier were NEVER called
        mock_retriever.retrieve.assert_not_called()
        mock_verifier.verify.assert_not_called()

        # Ensure gateway complete_stream was called once
        mock_gateway.complete_stream.assert_called_once()
        call_messages = mock_gateway.complete_stream.call_args[1]["messages"]
        self.assertEqual(call_messages[1]["content"], "Hello, who are you?")

    async def test_in_scope_paper_query_invokes_full_rag(self) -> None:
        """Verify in-scope paper question invokes retriever, answer generator, and citation verifier."""
        mock_retriever = MagicMock()
        mock_chunk = make_mock_chunk(paper_id="1706.03762", page_numbers=[4])
        mock_retriever.retrieve.return_value = [mock_chunk]

        mock_llm = MagicMock()
        rag_completion = CompletionResult(
            content="Scaled dot-product attention computes Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V [arXiv:1706.03762v7, p.4].",
            start_time=100.0,
            first_token_ms=50.0,
            total_ms=130.0,
            tokens_in=300,
            tokens_out=30,
            telemetry=RateLimitTelemetry(),
        )
        mock_llm.complete_stream = AsyncMock(return_value=rag_completion)

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_client=mock_llm,
        )

        result: PipelineResult = await orchestrator.run_pipeline("Explain scaled dot-product attention")

        # Assertions
        self.assertIn("step_2_retrieve", result.steps_executed)
        self.assertIn("step_3_draft", result.steps_executed)
        self.assertIn("step_4_verify", result.steps_executed)
        self.assertIn("accepted", result.steps_executed)
        self.assertEqual(len(result.chunks), 1)
        self.assertFalse(result.is_refusal)
        self.assertEqual(result.classification.intent, QueryIntent.CORPUS_RAG)

        # Ensure retriever was called
        mock_retriever.retrieve.assert_called_once_with(query="Explain scaled dot-product attention", top_k=None)

    async def test_out_of_scope_query_triggers_safe_refusal(self) -> None:
        """Verify out-of-scope non-corpus query triggers safe refusal without retrieval."""
        mock_retriever = MagicMock()
        mock_llm = MagicMock()

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            llm_client=mock_llm,
        )

        result: PipelineResult = await orchestrator.run_pipeline("How do I bake chocolate chip cookies?")

        self.assertTrue(result.is_refusal)
        self.assertEqual(result.steps_executed, ["step_1_analyze", "step_1_refusal"])
        self.assertEqual(result.answer, SAFE_OUT_OF_SCOPE_REFUSAL)
        self.assertEqual(len(result.chunks), 0)
        self.assertEqual(result.loop_count, 0)

        # Retrieval and LLM generation must be skipped
        mock_retriever.retrieve.assert_not_called()
        mock_llm.complete_stream.assert_not_called()

    @patch("tools.web_fetcher.fetch_web_page", new_callable=AsyncMock)
    async def test_external_web_routing_and_synthesis(self, mock_fetch: AsyncMock) -> None:
        """Verify EXTERNAL_WEB route invokes fetch_web_page and synthesizes with gateway."""
        mock_fetch.return_value = "Python 3.14 introduces improved performance and new typing features."

        mock_gateway = MagicMock()
        web_completion = CompletionResult(
            content="According to the web page, Python 3.14 introduces improved performance.",
            start_time=100.0,
            first_token_ms=55.0,
            total_ms=140.0,
            tokens_in=120,
            tokens_out=20,
            telemetry=RateLimitTelemetry(),
        )
        mock_gateway.complete_stream = AsyncMock(return_value=web_completion)
        mock_retriever = MagicMock()

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            gateway=mock_gateway,
        )

        result: PipelineResult = await orchestrator.run_pipeline("Fetch https://python.org and summarize it")

        # Assertions
        self.assertEqual(
            result.steps_executed,
            ["step_1_analyze", "external_web_fetch", "external_web_synthesize"],
        )
        self.assertEqual(result.classification.intent, QueryIntent.EXTERNAL_WEB)
        self.assertFalse(result.is_refusal)
        self.assertIn("Python 3.14", result.answer)

        # Web fetch called with correct URL
        mock_fetch.assert_called_once_with("https://python.org")

        # FAISS retriever never touched
        mock_retriever.retrieve.assert_not_called()

    async def test_direct_llm_uses_qwen_chat_model(self) -> None:
        """Verify DIRECT_LLM route defaults to qwen/qwen3.8-27b model for general conversation."""
        mock_retriever = MagicMock()
        mock_chat_gateway = MagicMock()
        mock_chat_gateway.model = "qwen/qwen3.8-27b"
        mock_chat_gateway.source = "real"

        direct_completion = CompletionResult(
            content="Hello from Qwen! I am ready to help.",
            start_time=100.0,
            first_token_ms=50.0,
            total_ms=130.0,
            tokens_in=25,
            tokens_out=15,
            telemetry=RateLimitTelemetry(),
        )
        mock_chat_gateway.complete_stream = AsyncMock(return_value=direct_completion)

        orchestrator = RAGOrchestrator(
            retriever=mock_retriever,
            chat_gateway=mock_chat_gateway,
        )

        result: PipelineResult = await orchestrator.run_pipeline("Hello, can you help me?")

        self.assertEqual(result.classification.intent, QueryIntent.DIRECT_LLM)
        self.assertEqual(result.telemetry["model"], "qwen/qwen3.8-27b")
        self.assertIn("Hello from Qwen", result.answer)
        mock_chat_gateway.complete_stream.assert_called_once()



if __name__ == "__main__":
    unittest.main()

