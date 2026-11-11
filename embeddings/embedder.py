"""Modular embedding generation using Sentence Transformers.

Processes text chunks into dense, normalized vectors with provenance tracking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from embeddings.metadata import (
    ChunkVectorMetadata,
    EmbeddingConfig,
    EmbeddingValidationReport,
    validate_embeddings,
)

logger = logging.getLogger(__name__)


class Embedder:
    """Manages dense vector generation using sentence-transformers."""

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        self.config = config or EmbeddingConfig()
        self._model = None

    def load_model(self):
        """Load the SentenceTransformer model lazily or explicitly."""
        if self._model is not None:
            return self._model

        logger.info("Loading embedding model: %s", self.config.model_name)
        try:
            from sentence_transformers import SentenceTransformer

            device = self.config.device
            self._model = SentenceTransformer(self.config.model_name, device=device)
            # Verify dimension
            if hasattr(self._model, "get_embedding_dimension"):
                dim = self._model.get_embedding_dimension()
            else:
                dim = self._model.get_sentence_embedding_dimension()
            if dim != self.config.expected_dimension:
                logger.warning(
                    "Model embedding dimension (%d) differs from expected_dimension (%d)",
                    dim,
                    self.config.expected_dimension,
                )
                self.config.expected_dimension = dim
            logger.info("Model loaded successfully. Embedding dimension: %d", dim)
            return self._model
        except Exception as exc:
            error_msg = f"Failed to load sentence-transformers model '{self.config.model_name}': {exc}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from exc

    def embed_texts(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
        normalize: Optional[bool] = None,
    ) -> np.ndarray:
        """Generate dense embeddings for a list of texts in batches.
        
        Args:
            texts: List of text strings to embed.
            batch_size: Override batch size.
            normalize: Override unit length normalization.
            
        Returns:
            np.ndarray of shape (len(texts), embedding_dim) as float32.
        """
        if not texts:
            return np.empty((0, self.config.expected_dimension), dtype=np.float32)

        model = self.load_model()
        bs = batch_size or self.config.batch_size
        norm = self.config.normalize_embeddings if normalize is None else normalize

        logger.info(
            "Generating embeddings for %d text(s) (batch_size=%d, normalize=%s)",
            len(texts),
            bs,
            norm,
        )

        try:
            embeddings = model.encode(
                texts,
                batch_size=bs,
                show_progress_bar=False,
                normalize_embeddings=norm,
                convert_to_numpy=True,
            )
            # Ensure float32
            embeddings = embeddings.astype(np.float32)

            # Extra safety check for normalization if requested
            if norm:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                # Replace zero norms with 1 to avoid division by zero
                norms = np.where(norms == 0, 1.0, norms)
                embeddings = embeddings / norms

            return embeddings
        except Exception as exc:
            error_msg = f"Embedding generation failed: {exc}"
            logger.error(error_msg, exc_info=True)
            raise RuntimeError(error_msg) from exc

    def embed_chunks(
        self,
        chunks: List[Dict[str, Any]],
        batch_size: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Convert a list of chunk dictionaries into normalized vectors and metadata.
        
        Args:
            chunks: List of chunk dictionaries from chunks.json.
            batch_size: Optional batch size override.
            
        Returns:
            (embeddings_array, metadata_list)
        """
        if not chunks:
            return (
                np.empty((0, self.config.expected_dimension), dtype=np.float32),
                [],
            )

        texts = [c.get("text", "") for c in chunks]
        embeddings = self.embed_texts(texts, batch_size=batch_size)

        dimension = int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.shape[0] > 0 else self.config.expected_dimension

        metadata_list: List[Dict[str, Any]] = []
        for i, c in enumerate(chunks):
            meta = ChunkVectorMetadata(
                vector_index=i,
                chunk_id=c.get("chunk_id", f"chunk_{i}"),
                paper_id=c.get("paper_id", "unknown"),
                title=c.get("title", ""),
                version=c.get("version"),
                source_filename=c.get("source_filename", ""),
                source_page_numbers=c.get("source_page_numbers", []),
                chunk_index=c.get("chunk_index", i + 1),
                token_count=c.get("token_count", 0),
                embedding_model=self.config.model_name,
                embedding_dimension=dimension,
            )
            metadata_list.append(meta.to_dict())

        return embeddings, metadata_list

    @staticmethod
    def save_embeddings(
        embeddings: np.ndarray,
        metadata: List[Dict[str, Any]],
        npy_path: Union[str, Path],
        meta_path: Union[str, Path],
        config: Optional[EmbeddingConfig] = None,
    ) -> None:
        """Save embeddings matrix (.npy) and provenance metadata (.json) to disk."""
        p_npy = Path(npy_path)
        p_meta = Path(meta_path)

        p_npy.parent.mkdir(parents=True, exist_ok=True)
        p_meta.parent.mkdir(parents=True, exist_ok=True)

        # Save numpy binary
        np.save(p_npy, embeddings)
        logger.info("Saved embeddings array to %s (shape: %s)", p_npy, embeddings.shape)

        # Save metadata JSON with top-level summary and chunk records
        meta_payload = {
            "model_name": config.model_name if config else "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
            "total_embeddings": int(embeddings.shape[0]) if embeddings.ndim == 2 else 0,
            "normalized": bool(config.normalize_embeddings if config else True),
            "embeddings": metadata,
        }

        with open(p_meta, "w", encoding="utf-8") as f:
            json.dump(meta_payload, f, indent=2, ensure_ascii=False)
        logger.info("Saved embedding metadata to %s (%d records)", p_meta, len(metadata))

    @staticmethod
    def load_embeddings(
        npy_path: Union[str, Path],
        meta_path: Union[str, Path],
    ) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
        """Load embeddings array and metadata from disk."""
        p_npy = Path(npy_path)
        p_meta = Path(meta_path)

        if not p_npy.exists():
            raise FileNotFoundError(f"Embeddings file not found: {p_npy}")
        if not p_meta.exists():
            raise FileNotFoundError(f"Embedding metadata file not found: {p_meta}")

        embeddings = np.load(p_npy)
        with open(p_meta, "r", encoding="utf-8") as f:
            payload = json.load(f)

        records = payload.get("embeddings", payload if isinstance(payload, list) else [])
        return embeddings, records, payload

