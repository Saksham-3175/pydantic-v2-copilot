"""Scoped retriever agent for fetching targeted documentation chunks from Qdrant vector database."""

from __future__ import annotations

import typing

from core_agents.schemas import DecomposedQuery, DocChunkPayload
from storage.qdrant_manager import QdrantStorageManager, ScoredChunk


class ScopedRetriever:
    """Agent that executes scoped retrieval against QdrantStorageManager based on DecomposedQuery."""

    def __init__(self, storage_manager: typing.Optional[QdrantStorageManager] = None):
        self.storage_manager = storage_manager

    async def retrieve(
        self,
        decomposed: DecomposedQuery,
        limit_per_topic: int = 2,
    ) -> list[DocChunkPayload]:
        """Fetch targeted documentation chunks with deduplication and edge case handling."""
        # Edge Case 1: Empty decomposed query (no topics or sub-queries)
        if not decomposed.detected_topics and not decomposed.sub_queries:
            return []

        if self.storage_manager is None:
            return []

        all_scored_chunks: list[ScoredChunk] = []

        # If topics exist, search per topic
        if decomposed.detected_topics:
            for i, topic in enumerate(decomposed.detected_topics):
                # Use corresponding sub_query if available, else use topic as query string
                query_str = (
                    decomposed.sub_queries[i]
                    if i < len(decomposed.sub_queries)
                    else f"how to migrate {topic}"
                )
                results = await self.storage_manager.search(
                    query=query_str,
                    topic=topic,
                    limit=limit_per_topic,
                )
                all_scored_chunks.extend(results)
        else:
            # Fallback: search globally across any sub_queries
            for query_str in decomposed.sub_queries:
                results = await self.storage_manager.search(
                    query=query_str,
                    topic=None,
                    limit=limit_per_topic,
                )
                all_scored_chunks.extend(results)

        # Edge Case 2: Deduplicate chunks by chunk_id
        seen_ids: set[str] = set()
        deduped_payloads: list[DocChunkPayload] = []

        for item in all_scored_chunks:
            if item.chunk.chunk_id not in seen_ids:
                seen_ids.add(item.chunk.chunk_id)
                deduped_payloads.append(
                    DocChunkPayload(
                        chunk_id=item.chunk.chunk_id,
                        topic=item.chunk.topic,
                        breadcrumb=item.chunk.breadcrumb,
                        content=item.chunk.payload_content,
                        score=item.score,
                    )
                )

        return deduped_payloads
