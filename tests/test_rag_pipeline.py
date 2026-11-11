"""Unit tests for the end-to-end RAG pipeline and prompt formatting."""

import unittest
from unittest.mock import AsyncMock, MagicMock

from llm.groq_client import CompletionResult, RateLimitTelemetry
from rag import RAGPipeline, format_context_prompt
from retrieval import RetrievedChunk, Retriever


class TestRAGPipeline(unittest.IsolatedAsyncioTestCase):
    """Test suite for context prompt assembly and end-to-end RAG pipeline flow."""

    def test_format_context_prompt(self) -> None:
        """Verify that context prompt formats reference passages and citations correctly."""
        chunks = [
            RetrievedChunk(
                chunk_id="chunk_1",
                paper_id="1706.03762",
                paper_title="Attention Is All You Need",
                version=7,
                page_numbers=[4, 5],
                text="Dot-product attention formula.",
                score=0.9,
                citation="[arXiv:1706.03762v7, pp.4, 5]",
            )
        ]
        prompt = format_context_prompt(query="Explain attention", chunks=chunks)
        self.assertIn("Reference Passage [1]", prompt)
        self.assertIn("[arXiv:1706.03762v7, pp.4, 5]", prompt)
        self.assertIn("Question: Explain attention", prompt)

    async def test_rag_pipeline_mocked_execution(self) -> None:
        """Verify full RAG pipeline execution with mocked LLM generation."""
        mock_chunks = [
            RetrievedChunk(
                chunk_id=f"chunk_{i}",
                paper_id="1706.03762",
                paper_title="Attention Is All You Need",
                version=7,
                page_numbers=[i],
                text=f"Passage text {i}",
                score=0.8 - i * 0.1,
                citation=f"[arXiv:1706.03762v7, p.{i}]",
            )
            for i in range(1, 4)
        ]

        # Mock retriever
        mock_retriever = MagicMock()
        mock_retriever.retrieve.return_value = mock_chunks

        # Mock LLM client
        mock_llm = MagicMock()
        mock_completion = CompletionResult(
            content="The Transformer utilizes multi-head attention [arXiv:1706.03762v7, p.1].",
            start_time=1000.0,
            first_token_ms=50.0,
            total_ms=120.0,
            tokens_in=350,
            tokens_out=25,
            telemetry=RateLimitTelemetry(),
        )
        mock_llm.complete_stream = AsyncMock(return_value=mock_completion)

        pipeline = RAGPipeline(
            retriever=mock_retriever,
            llm_client=mock_llm,
            setting="Setting A",
        )

        response = await pipeline.answer_query("What is Transformer?")

        self.assertEqual(len(response.retrieved_chunks), 3)
        self.assertIn("[arXiv:1706.03762v7, p.1]", response.citations)
        self.assertEqual(response.setting_name, "Setting A")
        self.assertGreater(response.retrieval_ms, 0.0)
        self.assertGreater(response.end_to_end_ms, 0.0)
        self.assertEqual(response.tokens_in, 350)
        self.assertEqual(response.tokens_out, 25)


if __name__ == "__main__":
    unittest.main()

