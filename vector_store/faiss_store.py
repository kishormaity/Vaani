"""FAISS Vector Store implementation using IndexFlatIP for cosine similarity search.

Provides deterministic mapping from FAISS vector positions to chunk metadata,
index serialization, reload, and semantic search capabilities.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import faiss
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Represents a single vector search match with similarity score and metadata."""

    vector_index: int
    similarity_score: float
    chunk_id: str
    paper_id: str
    title: str
    version: Optional[int]
    source_filename: str
    source_page_numbers: List[int]
    chunk_index: int
    token_count: int
    text: str = ""

    @property
    def chunk(self) -> SearchResult:
        """Enables compatibility with candidate.chunk.text access pattern."""
        return self

    def to_dict(self) -> Dict[str, Any]:
        return {
            "vector_index": self.vector_index,
            "similarity_score": round(self.similarity_score, 4),
            "chunk_id": self.chunk_id,
            "paper_id": self.paper_id,
            "title": self.title,
            "version": self.version,
            "source_filename": self.source_filename,
            "source_page_numbers": self.source_page_numbers,
            "chunk_index": self.chunk_index,
            "token_count": self.token_count,
            "text": self.text,
        }


class FAISSVectorStore:
    """Manages FAISS IndexFlatIP index with associated chunk metadata."""

    def __init__(
        self,
        dimension: int = 384,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        self.dimension = dimension
        self.model_name = model_name
        self.index: faiss.IndexFlatIP = faiss.IndexFlatIP(dimension)
        self.metadata_catalog: List[Dict[str, Any]] = []

    @property
    def total_vectors(self) -> int:
        """Return number of indexed vectors."""
        return self.index.ntotal

    def add_embeddings(
        self,
        embeddings: np.ndarray,
        metadata_list: List[Dict[str, Any]],
    ) -> None:
        """Add dense vectors and corresponding metadata to the index.
        
        Args:
            embeddings: 2D numpy array of float32 vectors.
            metadata_list: List of metadata dictionaries aligned 1-to-1 with rows.
        """
        if not isinstance(embeddings, np.ndarray):
            raise TypeError("Embeddings must be a numpy ndarray")

        if embeddings.ndim != 2:
            raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")

        num_vectors, dim = embeddings.shape
        if dim != self.dimension:
            raise ValueError(
                f"Embedding dimension ({dim}) does not match index dimension ({self.dimension})"
            )

        if len(metadata_list) != num_vectors:
            raise ValueError(
                f"Metadata count ({len(metadata_list)}) does not match vector count ({num_vectors})"
            )

        # Ensure float32 format
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)

        # Verify / enforce L2 normalization for cosine similarity
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        max_deviation = float(np.max(np.abs(norms - 1.0)))
        if max_deviation > 1e-3:
            logger.warning(
                "Vectors not unit-normalized (max deviation: %.4f); normalizing now",
                max_deviation,
            )
            norms = np.where(norms == 0, 1.0, norms)
            embeddings = embeddings / norms

        start_pos = self.index.ntotal
        self.index.add(embeddings)

        # Store aligned metadata
        for i, meta in enumerate(metadata_list):
            meta_copy = dict(meta)
            meta_copy["vector_index"] = start_pos + i
            self.metadata_catalog.append(meta_copy)

        logger.info(
            "Added %d vectors to FAISS index. Total indexed vectors: %d",
            num_vectors,
            self.index.ntotal,
        )

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_paper: Optional[str] = None,
    ) -> List[SearchResult]:
        """Search top_k nearest neighbors by cosine similarity.
        
        Args:
            query_vector: 1D or 2D vector for the query.
            top_k: Number of nearest neighbors to retrieve.
            filter_paper: Optional paper ID substring to filter candidates.
            
        Returns:
            List of SearchResult objects sorted by similarity score descending.
        """
        if self.index.ntotal == 0:
            return []

        q = np.array(query_vector, dtype=np.float32)
        if q.ndim == 1:
            q = np.expand_dims(q, axis=0)
        elif q.ndim != 2 or q.shape[0] != 1:
            raise ValueError(f"Query vector must be 1D or (1, dim), got {q.shape}")

        if q.shape[1] != self.dimension:
            raise ValueError(
                f"Query dimension ({q.shape[1]}) does not match index dimension ({self.dimension})"
            )

        # Normalize query vector to unit length for cosine similarity
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        k_search = min(self.index.ntotal, max(50, top_k * 10)) if filter_paper else min(top_k, self.index.ntotal)
        distances, indices = self.index.search(q, k_search)

        results: List[SearchResult] = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata_catalog):
                continue
            meta = self.metadata_catalog[idx]
            paper_id = meta.get("paper_id", "")
            if filter_paper and filter_paper not in paper_id:
                continue
            res = SearchResult(
                vector_index=idx,
                similarity_score=float(score),
                chunk_id=meta.get("chunk_id", f"chunk_{idx}"),
                paper_id=paper_id,
                title=meta.get("title", ""),
                version=meta.get("version"),
                source_filename=meta.get("source_filename", ""),
                source_page_numbers=meta.get("source_page_numbers", []),
                chunk_index=meta.get("chunk_index", idx + 1),
                token_count=meta.get("token_count", 0),
                text=meta.get("text", ""),
            )
            results.append(res)
            if len(results) >= top_k:
                break

        return results

    def save(
        self,
        index_path: Union[str, Path],
        catalog_path: Union[str, Path],
    ) -> None:
        """Serialize FAISS binary index and metadata catalog to disk."""
        p_index = Path(index_path)
        p_catalog = Path(catalog_path)

        p_index.parent.mkdir(parents=True, exist_ok=True)
        p_catalog.parent.mkdir(parents=True, exist_ok=True)

        # Write FAISS index
        faiss.write_index(self.index, str(p_index))
        logger.info("Saved FAISS index to %s (%d vectors)", p_index, self.index.ntotal)

        # Write catalog JSON
        catalog_payload = {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "metric": "cosine (IndexFlatIP with L2 normalized vectors)",
            "total_vectors": self.index.ntotal,
            "catalog": self.metadata_catalog,
        }
        with open(p_catalog, "w", encoding="utf-8") as f:
            json.dump(catalog_payload, f, indent=2, ensure_ascii=False)
        logger.info("Saved vector catalog to %s (%d records)", p_catalog, len(self.metadata_catalog))

    @classmethod
    def load(
        cls,
        index_path: Union[str, Path],
        catalog_path: Union[str, Path],
    ) -> FAISSVectorStore:
        """Load a persisted FAISS index and metadata catalog from disk."""
        p_index = Path(index_path)
        p_catalog = Path(catalog_path)

        if not p_index.exists():
            raise FileNotFoundError(f"FAISS index file not found: {p_index}")
        if not p_catalog.exists():
            raise FileNotFoundError(f"Vector catalog file not found: {p_catalog}")

        raw_index = faiss.read_index(str(p_index))
        with open(p_catalog, "r", encoding="utf-8") as f:
            catalog_payload = json.load(f)

        dimension = catalog_payload.get("dimension", raw_index.d)
        model_name = catalog_payload.get("model_name", "sentence-transformers/all-MiniLM-L6-v2")

        store = cls(dimension=dimension, model_name=model_name)
        store.index = raw_index
        store.metadata_catalog = catalog_payload.get("catalog", [])

        logger.info(
            "Loaded FAISSVectorStore from %s (%d vectors, model=%s)",
            p_index,
            store.index.ntotal,
            model_name,
        )
        return store

