"""Storage, hierarchical chunking, and Qdrant vector retrieval module."""

from storage.chunker import DocChunk, MarkdownHeaderChunker
from storage.embedder import (
    CachedEmbedder,
    DeterministicMockEmbedder,
    EmbedderProtocol,
    GeminiEmbedder,
    get_default_embedder,
)
from storage.ingest import (
    DEFAULT_COLLECTION_NAME,
    ensure_collection,
    ingest_corpus,
)
from storage.qdrant_manager import QdrantStorageManager, ScoredChunk

__all__ = [
    "DocChunk",
    "MarkdownHeaderChunker",
    "EmbedderProtocol",
    "DeterministicMockEmbedder",
    "GeminiEmbedder",
    "CachedEmbedder",
    "get_default_embedder",
    "DEFAULT_COLLECTION_NAME",
    "ensure_collection",
    "ingest_corpus",
    "QdrantStorageManager",
    "ScoredChunk",
]
