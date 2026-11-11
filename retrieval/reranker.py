"""Cross-Encoder re-ranking layer for dense vector retrieval candidates.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """Re-ranks retrieved candidates using a CrossEncoder transformer model."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def load_model(self):
        """Lazily load the CrossEncoder model."""
        if self._model is not None:
            return self._model

        logger.info("Loading CrossEncoder re-ranker model: %s", self.model_name)
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name, device=self.device)
            logger.info("CrossEncoder re-ranker loaded successfully.")
            return self._model
        except Exception as exc:
            error_msg = f"Failed to load CrossEncoder model '{self.model_name}': {exc}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from exc

    def rerank(
        self,
        query: str,
        candidates: List[Any],
        top_n: Optional[int] = None,
    ) -> List[Any]:
        """Re-rank candidate search results for a query.
        
        Args:
            query: The user search query string.
            candidates: List of SearchResult objects.
            top_n: Optional number of top results to return.
            
        Returns:
            List of SearchResult objects sorted descending by re-ranker score.
        """
        if not candidates:
            return []

        model = self.load_model()

        # Extract text safely supporting both candidate.chunk.text and candidate.text
        pairs: List[Tuple[str, str]] = []
        for cand in candidates:
            if hasattr(cand, "chunk") and hasattr(cand.chunk, "text"):
                text = cand.chunk.text
            elif hasattr(cand, "text"):
                text = cand.text
            else:
                text = str(cand)
            pairs.append((query, text))

        try:
            scores = model.predict(pairs)
        except Exception as exc:
            error_msg = f"CrossEncoder prediction failed: {exc}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from exc

        # Create updated candidate results with re-ranker score
        reranked_candidates = []
        for cand, score in zip(candidates, scores):
            cand_copy = copy.copy(cand)
            cand_copy.similarity_score = float(score)
            reranked_candidates.append(cand_copy)

        # Sort descending by re-ranker score
        reranked_candidates.sort(key=lambda x: x.similarity_score, reverse=True)

        if top_n is not None and top_n > 0:
            reranked_candidates = reranked_candidates[:top_n]

        return reranked_candidates

