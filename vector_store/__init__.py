"""Vector store package for The Relay.

Provides persistent FAISS IndexFlatIP vector storage and cosine similarity retrieval.
"""

from vector_store.faiss_store import FAISSVectorStore, SearchResult

__all__ = [
    "FAISSVectorStore",
    "SearchResult",
]

