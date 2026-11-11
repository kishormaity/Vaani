"""Citation Verifier for The Relay 4-step RAG pipeline.

Verifies that all citations produced in LLM answers:
1. Are present and formatted (e.g. [arXiv:1706.03762v7, p.5]).
2. Reference one of the three verified arXiv papers.
3. Cites arXiv IDs and page numbers that exist within the retrieved context chunks.
4. Detects and confirms legitimate refusal responses when context is insufficient.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# The three canonical arXiv paper IDs
VERIFIED_PAPER_IDS: Set[str] = {"1706.03762", "2410.00037", "2507.07955"}

# Common refusal language patterns
REFUSAL_PATTERNS = [
    r"cannot be found in the provided",
    r"not contain sufficient information",
    r"provided passages do not contain",
    r"do not contain (?:any|sufficient|this) information",
    r"does not contain (?:any|sufficient|this) information",
    r"no information provided in the (?:context|passages)",
    r"unable to (?:answer|find) (?:this|information)",
    r"i cannot answer based on the provided",
    r"not mentioned in the (?:context|reference)",
    r"do not contain sufficient evidence",
]


@dataclass
class VerificationResult:
    """Outcome of answer citation verification."""

    is_valid: bool
    is_refusal: bool
    valid_citations: List[str] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_valid": self.is_valid,
            "is_refusal": self.is_refusal,
            "valid_citations": self.valid_citations,
            "invalid_citations": self.invalid_citations,
            "error_message": self.error_message,
        }


# Backwards compatibility alias
ValidationResult = VerificationResult


class CitationVerifier:
    """Validates citations against retrieved context and allowed paper corpus."""

    def __init__(self, allowed_paper_ids: Optional[Set[str]] = None) -> None:
        self.allowed_paper_ids = allowed_paper_ids or VERIFIED_PAPER_IDS

    def parse_citation(self, citation_str: str) -> Tuple[Optional[str], List[int]]:
        """Parse paper ID and page numbers from a citation string.

        Examples:
            '[arXiv:1706.03762v7, p.5]' -> ('1706.03762', [5])
            '[arXiv:2410.00037v1, pp.2, 3]' -> ('2410.00037', [2, 3])
            '[arXiv:2507.07955, pp.1-3]' -> ('2507.07955', [1, 2, 3])
        """
        match = re.search(r"\[arXiv:\s*([^,\]]+)(?:,\s*([^\]]+))?\]", citation_str)
        if not match:
            return None, []

        raw_id = match.group(1).strip()
        base_id = re.sub(r"v\d+$", "", raw_id)

        pages: List[int] = []
        raw_pages = match.group(2)
        if raw_pages:
            cleaned_pages = re.sub(r"pp?\.\s*", "", raw_pages.strip())
            range_match = re.search(r"(\d+)\s*-\s*(\d+)", cleaned_pages)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                pages.extend(range(start, end + 1))
            else:
                for num_str in re.findall(r"\d+", cleaned_pages):
                    pages.append(int(num_str))

        return base_id, sorted(list(set(pages)))

    def verify(
        self,
        answer: str,
        retrieved_chunks: List[RetrievedChunk],
    ) -> VerificationResult:
        """Validate answer text against retrieved context chunks.

        Args:
            answer: Generated LLM response text.
            retrieved_chunks: Chunks passed in the prompt context.

        Returns:
            VerificationResult with validity, refusal flag, and error diagnostic.
        """
        # 1. Detect explicit refusal language
        lower_answer = answer.lower()
        for pattern in REFUSAL_PATTERNS:
            if re.search(pattern, lower_answer):
                return VerificationResult(
                    is_valid=True,
                    is_refusal=True,
                    valid_citations=[],
                    invalid_citations=[],
                    error_message=None,
                )

        # 2. Extract all citations matching [arXiv:...]
        citation_matches = re.findall(r"\[arXiv:[^\]]+\]", answer)
        if not citation_matches:
            return VerificationResult(
                is_valid=False,
                is_refusal=False,
                valid_citations=[],
                invalid_citations=[],
                error_message="No citations found in the response. All factual statements must cite source passages.",
            )

        # Build context page map: paper_id -> set of available page numbers
        context_page_map: Dict[str, Set[int]] = {}
        for chunk in retrieved_chunks:
            p_id = chunk.paper_id
            if p_id not in context_page_map:
                context_page_map[p_id] = set()
            context_page_map[p_id].update(chunk.page_numbers)

        valid_citations: List[str] = []
        invalid_citations: List[str] = []
        error_reasons: List[str] = []

        for cite_str in citation_matches:
            paper_id, cited_pages = self.parse_citation(cite_str)

            if not paper_id or paper_id not in self.allowed_paper_ids:
                invalid_citations.append(cite_str)
                error_reasons.append(
                    f"Citation '{cite_str}' refers to unknown or unverified paper '{paper_id}'."
                )
                continue

            if paper_id not in context_page_map:
                invalid_citations.append(cite_str)
                error_reasons.append(
                    f"Citation '{cite_str}' refers to paper '{paper_id}', which was not in the retrieved context."
                )
                continue

            if not cited_pages:
                invalid_citations.append(cite_str)
                error_reasons.append(
                    f"Citation '{cite_str}' does not specify page numbers (expected format: [arXiv:<id>, p.<num>])."
                )
                continue

            available_pages = context_page_map[paper_id]
            missing_pages = [p for p in cited_pages if p not in available_pages]
            if missing_pages:
                invalid_citations.append(cite_str)
                error_reasons.append(
                    f"Citation '{cite_str}' cites page(s) {missing_pages} of paper {paper_id}, "
                    f"but retrieved passages only contain page(s) {sorted(list(available_pages))}."
                )
            else:
                valid_citations.append(cite_str)

        if invalid_citations:
            combined_error = " ; ".join(error_reasons)
            return VerificationResult(
                is_valid=False,
                is_refusal=False,
                valid_citations=valid_citations,
                invalid_citations=invalid_citations,
                error_message=combined_error,
            )

        return VerificationResult(
            is_valid=True,
            is_refusal=False,
            valid_citations=valid_citations,
            invalid_citations=[],
            error_message=None,
        )

    # Alias validate to verify
    validate = verify


# Backwards compatibility alias
CitationValidator = CitationVerifier

