"""Configurable paragraph-aware text chunking pipeline for research papers.

Preserves paragraph boundaries, sentence integrity, token budgets,
and full source citation metadata (paper ID, title, version, and page numbers).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import tiktoken

from chunking.metadata import (
    ChunkConfig,
    ChunkMetadata,
    ChunkingError,
    ChunkingStatistics,
)

logger = logging.getLogger(__name__)

# Common academic/English abbreviations to avoid splitting false sentence boundaries
ABBREVIATION_PATTERN = re.compile(
    r"\b(et\s+al|e\.g|i\.e|fig|tab|eq|sec|ref|vol|no|vs|dr|prof|dept)\.$",
    re.IGNORECASE,
)

# Sentence boundary lookbehind regex: split after punctuation followed by whitespace and capital/quote/bracket
SENTENCE_SPLIT_REGEX = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'\(\[])')


class Tokenizer:
    """Wrapper around tiktoken with special-token safety and helper utilities."""

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding_name = encoding_name
        try:
            self._enc = tiktoken.get_encoding(encoding_name)
        except Exception:
            logger.warning(
                "Could not load encoding '%s' directly; falling back to cl100k_base",
                encoding_name,
            )
            self.encoding_name = "cl100k_base"
            self._enc = tiktoken.get_encoding("cl100k_base")

    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs safely without throwing on disallowed special tokens."""
        if not text:
            return []
        return self._enc.encode(text, disallowed_special=())

    def decode(self, tokens: List[int]) -> str:
        """Decode token IDs back to string."""
        return self._enc.decode(tokens)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in text."""
        if not text:
            return 0
        return len(self.encode(text))


@dataclass
class AtomicUnit:
    """An atomic text unit (sentence or sub-sentence) with provenance."""

    text: str
    tokens: int
    page_number: int
    prefix: str  # "\n\n" for paragraph boundary, " " for sentence boundary


class ParagraphAwareChunker:
    """Configurable paragraph-aware text chunker.
    
    Adheres to paragraph boundaries whenever possible, splitting oversized
    paragraphs at sentence boundaries, and sentences at word/token boundaries
    only when strictly necessary to respect chunk token budgets.
    """

    def __init__(self, config: Optional[ChunkConfig] = None) -> None:
        self.config = config or ChunkConfig()
        self.tokenizer = Tokenizer(self.config.tokenizer_name)

    def _split_paragraph_into_sentences(self, paragraph: str) -> List[str]:
        """Split a paragraph into sentences, preserving common abbreviations."""
        raw_parts = SENTENCE_SPLIT_REGEX.split(paragraph)
        merged: List[str] = []

        for part in raw_parts:
            part_clean = part.strip()
            if not part_clean:
                continue
            if merged and ABBREVIATION_PATTERN.search(merged[-1].strip()):
                merged[-1] = f"{merged[-1]} {part_clean}"
            else:
                merged.append(part_clean)

        return merged

    def _split_oversized_sentence(
        self, sentence: str, max_tokens: int, page_number: int, initial_prefix: str
    ) -> List[AtomicUnit]:
        """Safely split an oversized sentence that exceeds max_tokens."""
        units: List[AtomicUnit] = []
        words = sentence.split(" ")
        current_words: List[str] = []
        current_prefix = initial_prefix

        for word in words:
            candidate = " ".join(current_words + [word]).strip()
            if self.tokenizer.count_tokens(candidate) <= max_tokens:
                current_words.append(word)
            else:
                if current_words:
                    seg_text = " ".join(current_words)
                    units.append(
                        AtomicUnit(
                            text=seg_text,
                            tokens=self.tokenizer.count_tokens(seg_text),
                            page_number=page_number,
                            prefix=current_prefix,
                        )
                    )
                    current_words = [word]
                    current_prefix = " "
                else:
                    # Single word exceeds max_tokens: slice at token level
                    word_tokens = self.tokenizer.encode(word)
                    for start_t in range(0, len(word_tokens), max_tokens):
                        slice_toks = word_tokens[start_t : start_t + max_tokens]
                        slice_text = self.tokenizer.decode(slice_toks)
                        units.append(
                            AtomicUnit(
                                text=slice_text,
                                tokens=len(slice_toks),
                                page_number=page_number,
                                prefix=current_prefix,
                            )
                        )
                        current_prefix = " "

        if current_words:
            seg_text = " ".join(current_words)
            units.append(
                AtomicUnit(
                    text=seg_text,
                    tokens=self.tokenizer.count_tokens(seg_text),
                    page_number=page_number,
                    prefix=current_prefix,
                )
            )

        return units

    def split_pages_into_units(
        self, pages: List[Dict[str, Any]], max_tokens: int
    ) -> List[AtomicUnit]:
        """Convert page texts into a sequence of atomic units with paragraph and sentence preservation."""
        units: List[AtomicUnit] = []

        for page in pages:
            page_num = page["page_number"]
            page_text = page.get("text", "").strip()
            if not page_text:
                continue

            raw_paragraphs = page_text.split("\n\n")
            first_para_on_page = True

            for raw_para in raw_paragraphs:
                para = raw_para.strip()
                if not para:
                    continue

                para_prefix = "\n\n" if (units and not first_para_on_page) else ("\n\n" if units else "")
                first_para_on_page = False

                # Split paragraph into sentences
                sentences = self._split_paragraph_into_sentences(para)
                first_sent_in_para = True

                for sentence in sentences:
                    sentence = sentence.strip()
                    if not sentence:
                        continue

                    sent_prefix = para_prefix if first_sent_in_para else " "
                    first_sent_in_para = False

                    sent_tokens = self.tokenizer.count_tokens(sentence)
                    if sent_tokens <= max_tokens:
                        units.append(
                            AtomicUnit(
                                text=sentence,
                                tokens=sent_tokens,
                                page_number=page_num,
                                prefix=sent_prefix,
                            )
                        )
                    else:
                        # Sentence exceeds max_tokens: split into sub-segments
                        sub_units = self._split_oversized_sentence(
                            sentence=sentence,
                            max_tokens=max_tokens,
                            page_number=page_num,
                            initial_prefix=sent_prefix,
                        )
                        units.extend(sub_units)

        return units

    def chunk_paper(self, paper: Dict[str, Any]) -> List[ChunkMetadata]:
        """Chunk an extracted paper, preserving provenance and citation metadata.
        
        Args:
            paper: Paper dictionary containing paper_id, title, version, filename, and pages.
            
        Returns:
            List of ChunkMetadata objects.
        """
        paper_id = paper.get("paper_id", "unknown")
        title = paper.get("title", "Untitled")
        version = paper.get("version")
        filename = paper.get("filename", "")
        pages = paper.get("pages", [])

        if not pages:
            logger.warning("Paper '%s' has 0 pages; skipping chunking.", paper_id)
            return []

        units = self.split_pages_into_units(pages, max_tokens=self.config.chunk_size)
        if not units:
            logger.warning("No text units generated for paper '%s'.", paper_id)
            return []

        chunks: List[ChunkMetadata] = []
        total_units = len(units)
        start_idx = 0
        chunk_index = 1  # 1-indexed chunk counter per paper

        while start_idx < total_units:
            chunk_units: List[AtomicUnit] = []
            chunk_text = ""
            current_tokens = 0
            end_idx = start_idx

            for i in range(start_idx, total_units):
                u = units[i]
                delimiter = u.prefix if chunk_text else ""
                candidate_text = f"{chunk_text}{delimiter}{u.text}"
                candidate_tokens = self.tokenizer.count_tokens(candidate_text)

                if candidate_tokens <= self.config.chunk_size or not chunk_units:
                    chunk_units.append(u)
                    chunk_text = candidate_text
                    current_tokens = candidate_tokens
                    end_idx = i
                else:
                    break

            if not chunk_units:
                break

            # Deduplicate and sort source page numbers
            source_pages = sorted(list({u.page_number for u in chunk_units}))

            chunk_id = f"{paper_id}_c{chunk_index:04d}"
            chunk_meta = ChunkMetadata(
                chunk_id=chunk_id,
                paper_id=paper_id,
                title=title,
                version=version,
                source_filename=filename,
                source_page_numbers=source_pages,
                chunk_index=chunk_index,
                token_count=current_tokens,
                text=chunk_text,
            )
            chunks.append(chunk_meta)
            chunk_index += 1

            # If this chunk consumed the final unit, we are completely finished
            if end_idx >= total_units - 1:
                break

            # Calculate overlap for next chunk
            overlap_tokens = 0
            overlap_start = end_idx + 1

            for j in range(end_idx, start_idx, -1):
                u_tok = units[j].tokens
                if overlap_tokens + u_tok <= self.config.chunk_overlap:
                    overlap_tokens += u_tok
                    overlap_start = j
                else:
                    # If no units fit under overlap, include at least 1 unit if within 1.3x overlap
                    if overlap_tokens == 0 and u_tok <= int(self.config.chunk_overlap * 1.3):
                        overlap_tokens += u_tok
                        overlap_start = j
                    break

            # Ensure forward progress to prevent infinite loops or duplicate-only chunks
            next_start_idx = max(overlap_start, start_idx + 1)
            if next_start_idx > end_idx + 1:
                next_start_idx = end_idx + 1

            start_idx = next_start_idx

        return chunks

    def chunk_dataset(
        self, papers_payload: Dict[str, Any]
    ) -> Tuple[List[ChunkMetadata], ChunkingStatistics]:
        """Process an entire dataset of papers and compute aggregate statistics.
        
        Args:
            papers_payload: Dictionary loaded from papers.json.
            
        Returns:
            (all_chunks, statistics)
        """
        papers = papers_payload.get("papers", [])
        all_chunks: List[ChunkMetadata] = []
        chunks_per_paper: Dict[str, int] = {}
        total_source_pages = 0

        for paper in papers:
            p_id = paper.get("paper_id", "unknown")
            total_source_pages += paper.get("total_pages", len(paper.get("pages", [])))
            paper_chunks = self.chunk_paper(paper)
            chunks_per_paper[p_id] = len(paper_chunks)
            all_chunks.extend(paper_chunks)
            logger.info(
                "Paper '%s' chunked: %d chunks generated across %d pages",
                p_id,
                len(paper_chunks),
                len(paper.get("pages", [])),
            )

        token_counts = [c.token_count for c in all_chunks]
        avg_tokens = (sum(token_counts) / len(token_counts)) if token_counts else 0.0
        min_tokens = min(token_counts) if token_counts else 0
        max_tokens = max(token_counts) if token_counts else 0
        exceeding = sum(1 for tc in token_counts if tc > self.config.chunk_size)

        stats = ChunkingStatistics(
            total_papers=len(papers),
            total_source_pages=total_source_pages,
            total_chunks=len(all_chunks),
            chunks_per_paper=chunks_per_paper,
            average_tokens_per_chunk=avg_tokens,
            min_tokens_per_chunk=min_tokens,
            max_tokens_per_chunk=max_tokens,
            chunks_exceeding_limit=exceeding,
        )

        return all_chunks, stats

