"""Context Sufficiency Checker for The Relay RAG pipeline.

Evaluates retrieved candidate chunks from Retriever to verify that context
relevance and depth are sufficient to warrant LLM answer generation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@dataclass
class ContextCheckerConfig:
    """Thresholds for context sufficiency evaluation."""

    min_chunks: int = 1
    # Minimum top score for FAISS cosine similarity (typically 0.0 - 1.0)
    min_dense_score: float = 0.20
    # Minimum top score for Cross-Encoder logits (typically -11.0 to +10.0)
    min_cross_encoder_score: float = -8.0


@dataclass
class ContextCheckResult:
    """Outcome of context sufficiency evaluation."""

    is_sufficient: bool
    num_chunks: int
    top_score: float
    avg_score: float
    reasoning: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_sufficient": self.is_sufficient,
            "num_chunks": self.num_chunks,
            "top_score": round(self.top_score, 4),
            "avg_score": round(self.avg_score, 4),
            "reasoning": self.reasoning,
        }


class ContextChecker:
    """Evaluates relevance and sufficiency of retrieved research paper passages."""

    def __init__(self, config: Optional[ContextCheckerConfig] = None) -> None:
        self.config = config or ContextCheckerConfig()

    def check_context(
        self,
        chunks: List[RetrievedChunk],
        is_cross_encoded: bool = False,
    ) -> ContextCheckResult:
        """Check if retrieved chunks are sufficient to answer a query.

        Args:
            chunks: List of RetrievedChunk objects from Retriever.
            is_cross_encoded: True if scores are cross-encoder logits, False for FAISS cosine similarities.

        Returns:
            ContextCheckResult indicating sufficiency, score telemetry, and reasoning.
        """
        if not chunks or len(chunks) < self.config.min_chunks:
            return ContextCheckResult(
                is_sufficient=False,
                num_chunks=0,
                top_score=0.0,
                avg_score=0.0,
                reasoning="Retriever returned no candidate chunks.",
            )

        scores = [chunk.score for chunk in chunks]
        top_score = max(scores)
        avg_score = sum(scores) / len(scores)

        # Detect score scale if not explicitly told
        # FAISS cosine scores on unit vectors are in [-1.0, 1.0], usually [0.2, 0.9]
        # CrossEncoder logits can be < -1.0 or > 1.0
        score_threshold = (
            self.config.min_cross_encoder_score
            if (is_cross_encoded or top_score < -1.0 or top_score > 1.0)
            else self.config.min_dense_score
        )

        if top_score < score_threshold:
            return ContextCheckResult(
                is_sufficient=False,
                num_chunks=len(chunks),
                top_score=top_score,
                avg_score=avg_score,
                reasoning=(
                    f"Top retrieval score ({top_score:.4f}) is below sufficiency threshold "
                    f"({score_threshold:.4f}). Insufficient evidence to formulate an answer."
                ),
            )

        return ContextCheckResult(
            is_sufficient=True,
            num_chunks=len(chunks),
            top_score=top_score,
            avg_score=avg_score,
            reasoning=(
                f"Context verified: {len(chunks)} chunks retrieved with top relevance score "
                f"{top_score:.4f} (threshold {score_threshold:.4f})."
            ),
        )

