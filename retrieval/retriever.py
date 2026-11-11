"""Modular Retriever integrating FAISS vector search and CrossEncoder re-ranking.

Implements the preset retrieval configurations for the Table 3 experiments:
- Setting A: top_k=3, use_reranker=False
- Setting B: top_k=5, use_reranker=False
- Setting C: candidate_k=8, top_k=3, use_reranker=True
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from embeddings import Embedder
from retrieval.reranker import CrossEncoderReranker
from vector_store import FAISSVectorStore, SearchResult

logger = logging.getLogger(__name__)

# Standard preset configurations required by the Table 3 experimental brief
PRESET_CONFIGURATIONS: Dict[str, Dict[str, Any]] = {
    "Setting A": {
        "top_k": 3,
        "use_reranker": False,
        "candidate_k": 3,
        "description": "Dense FAISS retrieval only (top_k=3)",
    },
    "Setting B": {
        "top_k": 5,
        "use_reranker": False,
        "candidate_k": 5,
        "description": "Dense FAISS retrieval only (top_k=5)",
    },
    "Setting C": {
        "candidate_k": 8,
        "top_k": 3,
        "use_reranker": True,
        "description": "Dense FAISS retrieval (candidate_k=8) + CrossEncoder re-ranking (top_k=3)",
    },
}


def format_citation(paper_id: str, version: Optional[int], page_numbers: List[int]) -> str:
    """Format academic citation string according to project convention.
    
    Example: [arXiv:1706.03762v7, p.5] or [arXiv:1706.03762v7, pp.1, 2]
    """
    ver_str = f"v{version}" if version is not None else ""
    if not page_numbers:
        page_str = "p.?"
    elif len(page_numbers) == 1:
        page_str = f"p.{page_numbers[0]}"
    else:
        page_str = f"pp.{', '.join(str(p) for p in page_numbers)}"

    return f"[arXiv:{paper_id}{ver_str}, {page_str}]"


@dataclass
class RetrievedChunk:
    """Represents a retrieved research paper chunk with provenance and formatted citation."""

    chunk_id: str
    paper_id: str
    paper_title: str
    version: Optional[int]
    page_numbers: List[int]
    text: str
    score: float
    citation: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "paper_title": self.paper_title,
            "version": self.version,
            "page_numbers": self.page_numbers,
            "text": self.text,
            "score": round(self.score, 4),
            "citation": self.citation,
        }


class Retriever:
    """Modular retriever combining dense vector search and cross-encoder re-ranking."""

    def __init__(
        self,
        vector_store: Optional[FAISSVectorStore] = None,
        embedder: Optional[Embedder] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        setting: Optional[str] = None,
        top_k: int = 3,
        candidate_k: int = 8,
        use_reranker: bool = False,
        index_path: Union[str, Path] = "data/vector_store/faiss.index",
        catalog_path: Union[str, Path] = "data/vector_store/chunk_catalog.json",
    ) -> None:
        self.index_path = Path(index_path)
        self.catalog_path = Path(catalog_path)
        self._vector_store = vector_store
        self._embedder = embedder
        self._reranker = reranker

        # Apply preset configuration if provided
        if setting:
            if setting not in PRESET_CONFIGURATIONS:
                raise ValueError(
                    f"Unknown preset '{setting}'. Available presets: {list(PRESET_CONFIGURATIONS.keys())}"
                )
            preset = PRESET_CONFIGURATIONS[setting]
            self.setting_name = setting
            self.top_k = preset["top_k"]
            self.candidate_k = preset["candidate_k"]
            self.use_reranker = preset["use_reranker"]
        else:
            self.setting_name = "Custom"
            self.top_k = top_k
            self.candidate_k = candidate_k
            self.use_reranker = use_reranker

    @property
    def vector_store(self) -> FAISSVectorStore:
        """Lazily load the FAISS vector store and catalog if not provided."""
        if self._vector_store is None:
            logger.info("Loading FAISSVectorStore from %s and %s", self.index_path, self.catalog_path)
            self._vector_store = FAISSVectorStore.load(
                index_path=self.index_path,
                catalog_path=self.catalog_path,
            )
        return self._vector_store

    @property
    def embedder(self) -> Embedder:
        """Lazily initialize the query Embedder."""
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    @property
    def reranker(self) -> CrossEncoderReranker:
        """Lazily initialize the CrossEncoderReranker."""
        if self._reranker is None:
            self._reranker = CrossEncoderReranker()
        return self._reranker

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_paper: Optional[str] = None,
    ) -> List[RetrievedChunk]:
        """Execute retrieval pipeline for the input query according to active configuration.
        
        Args:
            query: User search query string.
            top_k: Optional override for number of final returned chunks.
            filter_paper: Optional paper ID substring to filter candidates.
            
        Returns:
            List of RetrievedChunk objects with scores and formatted citations.
        """
        k_final = top_k or self.top_k

        # 1. Embed query
        query_vector = self.embedder.embed_texts([query])[0]

        # 2. Stage 1: Dense FAISS retrieval
        if self.use_reranker:
            k_initial = max(self.candidate_k, k_final)
            raw_candidates = self.vector_store.search(
                query_vector=query_vector,
                top_k=k_initial,
                filter_paper=filter_paper,
            )

            # Stage 2: Cross-Encoder re-ranking
            ranked_results = self.reranker.rerank(
                query=query,
                candidates=raw_candidates,
                top_n=k_final,
            )
        else:
            ranked_results = self.vector_store.search(
                query_vector=query_vector,
                top_k=k_final,
                filter_paper=filter_paper,
            )

        # 3. Format into structured RetrievedChunk objects
        output_chunks: List[RetrievedChunk] = []
        for res in ranked_results:
            citation_str = format_citation(
                paper_id=res.paper_id,
                version=res.version,
                page_numbers=res.source_page_numbers,
            )
            item = RetrievedChunk(
                chunk_id=res.chunk_id,
                paper_id=res.paper_id,
                paper_title=res.title,
                version=res.version,
                page_numbers=res.source_page_numbers,
                text=res.text,
                score=float(res.similarity_score),
                citation=citation_str,
            )
            output_chunks.append(item)

        return output_chunks

