"""Scoped vector retrieval and automatic fallback manager for Qdrant vector database."""

from dataclasses import dataclass
from pathlib import Path
import typing

from qdrant_client import AsyncQdrantClient, models

from storage.chunker import DocChunk
from storage.embedder import EmbedderProtocol, get_default_embedder
from storage.ingest import DEFAULT_COLLECTION_NAME


@dataclass
class ScoredChunk:
    """Represents a retrieved DocChunk paired with similarity score and fallback status."""

    chunk: DocChunk
    score: float
    fallback_triggered: bool = False


class QdrantStorageManager:
    """Manages scoped vector retrieval, automatic fallback, and AsyncQdrantClient lifecycle."""

    def __init__(
        self,
        client: AsyncQdrantClient,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedder: typing.Optional[EmbedderProtocol] = None,
        fallback_threshold: float = 0.55,
    ) -> None:
        self.client = client
        self.collection_name = collection_name
        self.embedder: EmbedderProtocol = (
            embedder if embedder is not None else get_default_embedder()
        )
        self.fallback_threshold = fallback_threshold

    async def search(
        self,
        query: str,
        topic: typing.Optional[str] = None,
        limit: int = 3,
        score_threshold: typing.Optional[float] = None,
    ) -> list[ScoredChunk]:
        """Perform scoped semantic search with automatic fallback to global retrieval.

        1. Generates query vector: `query_vector = await self.embedder.embed_query(query)`
        2. If `topic` is provided:
           - Constructs filter on topic payload field.
           - Queries Qdrant collection via `query_points`.
           - Fallback rule: if not res.points or res.points[0].score < self.fallback_threshold:
             - Sets fallback_triggered = True
             - Re-queries globally without filter.
        3. If `topic` is None:
           - Queries Qdrant globally with fallback_triggered = False.
        4. Reconstructs DocChunk from point.payload and returns list of ScoredChunk.
        """
        query_vector = await self.embedder.embed_query(query)
        fallback_triggered = False

        if topic is not None:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="topic",
                        match=models.MatchValue(value=topic),
                    )
                ]
            )
            res = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=query_filter,
                limit=limit,
                score_threshold=score_threshold,
            )
            if not res.points or res.points[0].score < self.fallback_threshold:
                fallback_triggered = True
                res = await self.client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=None,
                    limit=limit,
                    score_threshold=score_threshold,
                )
        else:
            res = await self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=None,
                limit=limit,
                score_threshold=score_threshold,
            )

        scored_chunks: list[ScoredChunk] = []
        for point in res.points:
            payload = point.payload or {}
            doc_chunk = DocChunk(
                chunk_id=payload.get("chunk_id", ""),
                topic=payload.get("topic", ""),
                source_file=payload.get("source_file", ""),
                breadcrumb=payload.get("breadcrumb", ""),
                search_text="",
                payload_content=payload.get("content", ""),
            )
            scored_chunks.append(
                ScoredChunk(
                    chunk=doc_chunk,
                    score=point.score,
                    fallback_triggered=fallback_triggered,
                )
            )

        return scored_chunks

    async def __aenter__(self) -> "QdrantStorageManager":
        return self

    async def __aexit__(
        self,
        exc_type: typing.Optional[typing.Type[BaseException]],
        exc_val: typing.Optional[BaseException],
        exc_tb: typing.Optional[typing.Any],
    ) -> None:
        await self.close()

    async def close(self) -> None:
        """Close underlying AsyncQdrantClient connection."""
        await self.client.close()
