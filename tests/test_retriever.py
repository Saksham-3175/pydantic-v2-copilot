"""Unit and integration tests for ScopedRetriever agent."""

from pathlib import Path
import pytest
from qdrant_client import AsyncQdrantClient

from core_agents import DecomposedQuery, DocChunkPayload, ScopedRetriever
from storage.embedder import CachedEmbedder
from storage.ingest import DEFAULT_COLLECTION_NAME, ingest_corpus
from storage.qdrant_manager import QdrantStorageManager

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "storage" / "data"
CACHE_FILE_PATH = DATA_DIR / "gemini_embeddings_cache.json"


async def create_test_storage_manager(
    fallback_threshold: float = 0.55,
) -> QdrantStorageManager:
    """Helper to populate an in-memory Qdrant instance and return a configured QdrantStorageManager."""
    client = AsyncQdrantClient(":memory:")
    embedder = CachedEmbedder(cache_path=CACHE_FILE_PATH)
    await ingest_corpus(
        data_dir=DATA_DIR,
        client=client,
        embedder=embedder,
        collection_name=DEFAULT_COLLECTION_NAME,
        recreate=True,
    )
    return QdrantStorageManager(
        client=client,
        collection_name=DEFAULT_COLLECTION_NAME,
        embedder=embedder,
        fallback_threshold=fallback_threshold,
    )


@pytest.mark.asyncio
async def test_retriever_happy_path():
    """Queries with topics ['pydantic_validators']; asserts returned chunks have chunk_id='pydantic_v2_migration#field-validation'."""
    manager = await create_test_storage_manager()
    try:
        retriever = ScopedRetriever(storage_manager=manager)
        decomposed = DecomposedQuery(
            summary="Migrate legacy @validator",
            detected_topics=["pydantic_validators"],
            sub_queries=["replace @validator with modern field validator"],
        )

        results = await retriever.retrieve(decomposed, limit_per_topic=2)

        assert len(results) > 0
        assert any(
            chunk.chunk_id == "pydantic_v2_migration#field-validation"
            for chunk in results
        )
        assert results[0].chunk_id == "pydantic_v2_migration#field-validation"
        assert isinstance(results[0], DocChunkPayload)
        assert results[0].topic == "pydantic_validators"
        assert results[0].score > 0.0
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_retriever_edge_case_empty_decomposed_query():
    """Feeds DecomposedQuery(summary='none', detected_topics=[], sub_queries=[]). Asserts returns [] without error."""
    manager = await create_test_storage_manager()
    try:
        retriever = ScopedRetriever(storage_manager=manager)
        decomposed = DecomposedQuery(
            summary="none",
            detected_topics=[],
            sub_queries=[],
        )

        results = await retriever.retrieve(decomposed)

        assert results == []
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_retriever_edge_case_deduplication():
    """Passes multiple sub-queries that match the same chunk; asserts len(results) == 1 and chunk appears only once."""
    manager = await create_test_storage_manager()
    try:
        retriever = ScopedRetriever(storage_manager=manager)
        # Two sub-queries targeting the exact same validator chunk with limit_per_topic=1
        decomposed = DecomposedQuery(
            summary="Migrate legacy @validator",
            detected_topics=[],
            sub_queries=[
                "replace @validator with modern field validator",
                "replace @validator with modern field validator",
            ],
        )

        results = await retriever.retrieve(decomposed, limit_per_topic=1)

        assert len(results) == 1
        assert results[0].chunk_id == "pydantic_v2_migration#field-validation"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_retriever_edge_case_unknown_topic_triggers_fallback():
    """Passes an unknown topic topic='unknown_topic'. Asserts Qdrant manager triggers fallback and returns relevant global chunks without crashing."""
    manager = await create_test_storage_manager()
    try:
        retriever = ScopedRetriever(storage_manager=manager)
        decomposed = DecomposedQuery(
            summary="Model configuration with unknown topic",
            detected_topics=["unknown_topic"],
            sub_queries=["how to configure model with ConfigDict"],
        )

        results = await retriever.retrieve(decomposed, limit_per_topic=2)

        assert len(results) > 0
        assert results[0].chunk_id == "pydantic_v2_migration#model-configuration"
        assert results[0].topic == "pydantic_config"
    finally:
        await manager.close()
