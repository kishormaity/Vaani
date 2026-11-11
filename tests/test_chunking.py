"""Unit tests for the configurable, paragraph-aware chunking pipeline."""

import unittest
from pathlib import Path

from chunking import (
    ChunkConfig,
    ChunkMetadata,
    ParagraphAwareChunker,
    Tokenizer,
    validate_chunks,
)


class TestParagraphAwareChunker(unittest.TestCase):
    """Test suite covering paragraph-aware chunking, token budgets, and metadata preservation."""

    def setUp(self) -> None:
        self.config = ChunkConfig(chunk_size=100, chunk_overlap=25, tokenizer_name="cl100k_base")
        self.chunker = ParagraphAwareChunker(self.config)
        self.tokenizer = Tokenizer("cl100k_base")

    def test_empty_input(self) -> None:
        """Verify handling of empty paper input and pages."""
        paper_empty_pages = {
            "paper_id": "test_empty",
            "title": "Empty Test",
            "version": 1,
            "filename": "empty.pdf",
            "pages": [],
        }
        chunks = self.chunker.chunk_paper(paper_empty_pages)
        self.assertEqual(chunks, [])

        paper_blank_text = {
            "paper_id": "test_blank",
            "title": "Blank Test",
            "version": 1,
            "filename": "blank.pdf",
            "pages": [{"page_number": 1, "text": "   \n\n   \t  "}],
        }
        chunks_blank = self.chunker.chunk_paper(paper_blank_text)
        self.assertEqual(chunks_blank, [])

    def test_one_short_paragraph(self) -> None:
        """Verify that a single short paragraph produces exactly one chunk."""
        text = "This is a brief introductory paragraph of research text. It contains two short sentences."
        paper = {
            "paper_id": "short_paper",
            "title": "Short Paper",
            "version": 1,
            "filename": "short.pdf",
            "pages": [{"page_number": 1, "text": text}],
        }
        chunks = self.chunker.chunk_paper(paper)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_id, "short_paper_c0001")
        self.assertEqual(chunks[0].chunk_index, 1)
        self.assertEqual(chunks[0].source_page_numbers, [1])
        self.assertEqual(chunks[0].text, text)
        self.assertTrue(chunks[0].token_count <= self.config.chunk_size)

    def test_multiple_paragraphs_combined(self) -> None:
        """Verify that multiple small paragraphs are combined with preserved paragraph boundaries."""
        para1 = "Paragraph one introduces the neural architecture."
        para2 = "Paragraph two describes the self-attention mechanism."
        para3 = "Paragraph three details the experimental setup."
        combined_text = f"{para1}\n\n{para2}\n\n{para3}"

        paper = {
            "paper_id": "multi_para",
            "title": "Multi Para",
            "version": 1,
            "filename": "multi.pdf",
            "pages": [{"page_number": 1, "text": combined_text}],
        }
        chunks = self.chunker.chunk_paper(paper)
        # All three paragraphs together are ~30 tokens, fitting easily in 100-token chunk
        self.assertEqual(len(chunks), 1)
        self.assertIn("\n\n", chunks[0].text)
        self.assertEqual(chunks[0].text, combined_text)

    def test_oversized_paragraph_split_at_sentence_boundaries(self) -> None:
        """Verify that an oversized paragraph is split at sentence boundaries without exceeding token limit."""
        # Create a paragraph with 15 sentences, total ~200 tokens (> 100 token chunk_size)
        sentences = [
            f"Sentence number {i} discusses specific theoretical aspects of transformer layers."
            for i in range(1, 16)
        ]
        long_para = " ".join(sentences)

        paper = {
            "paper_id": "long_para_paper",
            "title": "Long Para Paper",
            "version": 1,
            "filename": "long.pdf",
            "pages": [{"page_number": 1, "text": long_para}],
        }
        chunks = self.chunker.chunk_paper(paper)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c.token_count, self.config.chunk_size)
            # Verify each chunk contains non-empty text and valid metadata
            self.assertTrue(len(c.text.strip()) > 0)

    def test_overlap_behavior(self) -> None:
        """Verify that consecutive chunks have overlap without duplicating chunks or looping."""
        sentences = [
            f"Sentence {i} gives distinct contextual information for empirical verification."
            for i in range(1, 25)
        ]
        text = " ".join(sentences)

        paper = {
            "paper_id": "overlap_paper",
            "title": "Overlap Paper",
            "version": 1,
            "filename": "overlap.pdf",
            "pages": [{"page_number": 1, "text": text}],
        }
        chunks = self.chunker.chunk_paper(paper)
        self.assertGreater(len(chunks), 1)

        for i in range(len(chunks) - 1):
            c1 = chunks[i]
            c2 = chunks[i + 1]
            # Ensure chunks are not identical
            self.assertNotEqual(c1.text, c2.text)
            # Forward progress
            self.assertEqual(c2.chunk_index, c1.chunk_index + 1)
            # Overlap exists: end of c1 shares text with start of c2
            last_sentence_c1 = c1.text.split(".")[-2].strip() + "."
            self.assertIn(last_sentence_c1, c2.text)

    def test_metadata_preservation(self) -> None:
        """Verify that all required metadata fields are accurately preserved."""
        paper = {
            "paper_id": "1706.03762",
            "title": "Attention Is All You Need",
            "version": 7,
            "filename": "Attention_Is_All_You_Need_v7.pdf",
            "pages": [
                {"page_number": 1, "text": "Page one text content."},
                {"page_number": 2, "text": "Page two text content."},
            ],
        }
        chunks = self.chunker.chunk_paper(paper)
        self.assertGreater(len(chunks), 0)

        required_keys = {
            "chunk_id",
            "paper_id",
            "title",
            "version",
            "source_filename",
            "source_page_numbers",
            "chunk_index",
            "token_count",
            "text",
        }
        for c in chunks:
            d = c.to_dict()
            self.assertEqual(required_keys - set(d.keys()), set())
            self.assertEqual(c.paper_id, "1706.03762")
            self.assertEqual(c.version, 7)
            self.assertEqual(c.source_filename, "Attention_Is_All_You_Need_v7.pdf")
            self.assertEqual(c.token_count, self.tokenizer.count_tokens(c.text))

    def test_unique_chunk_ids(self) -> None:
        """Verify that every generated chunk has a distinct, stable chunk ID."""
        paper1 = {
            "paper_id": "paper_A",
            "title": "Paper A",
            "version": 1,
            "filename": "A.pdf",
            "pages": [{"page_number": 1, "text": "Text for paper A sentence one. Text sentence two."}],
        }
        paper2 = {
            "paper_id": "paper_B",
            "title": "Paper B",
            "version": 1,
            "filename": "B.pdf",
            "pages": [{"page_number": 1, "text": "Text for paper B sentence one. Text sentence two."}],
        }
        payload = {"papers": [paper1, paper2]}
        chunks, _ = self.chunker.chunk_dataset(payload)

        chunk_ids = [c.chunk_id for c in chunks]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)))
        self.assertEqual(chunks[0].chunk_id, "paper_A_c0001")
        self.assertEqual(chunks[1].chunk_id, "paper_B_c0001")

    def test_page_traceability(self) -> None:
        """Verify that chunks spanning multiple pages trace every represented page number accurately."""
        # Page 1 has 30 tokens, Page 2 has 30 tokens -> combined fits into 100 token chunk
        page1_text = "Content from the first physical page of the document."
        page2_text = "Content from the second physical page of the document."

        paper = {
            "paper_id": "trace_paper",
            "title": "Trace Paper",
            "version": 1,
            "filename": "trace.pdf",
            "pages": [
                {"page_number": 1, "text": page1_text},
                {"page_number": 2, "text": page2_text},
            ],
        }
        chunks = self.chunker.chunk_paper(paper)
        self.assertEqual(len(chunks), 1)
        # Should record both source pages [1, 2]
        self.assertEqual(chunks[0].source_page_numbers, [1, 2])
        self.assertIn("first physical page", chunks[0].text)
        self.assertIn("second physical page", chunks[0].text)


if __name__ == "__main__":
    unittest.main()

