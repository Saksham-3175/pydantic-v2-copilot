"""Integration and unit tests for QdrantStorageManager scoped vector retrieval and automatic fallback."""

from pathlib import Path
import pytest
from qdrant_client import AsyncQdrantClient

from storage import DocChunk, QdrantStorageManager, ScoredChunk
from storage.embedder import CachedEmbedder
from storage.ingest import DEFAULT_COLLECTION_NAME, ingest_corpus

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
async def test_scoped_search_matching_topic():
    """Test 1: Query about validators with topic='pydantic_validators' returns results where all chunks have topic == 'pydantic_validators' and fallback_triggered == False."""
    manager = await create_test_storage_manager()
    try:
        query = "replace @validator with modern field validator"
        results = await manager.search(
            query=query,
            topic="pydantic_validators",
            limit=3,
        )

        assert len(results) > 0
        assert all(r.chunk.topic == "pydantic_validators" for r in results)
        assert all(not r.fallback_triggered for r in results)
        assert results[0].chunk.chunk_id == "pydantic_v2_migration#field-validation"
        assert results[0].score >= manager.fallback_threshold
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_automatic_fallback_on_mismatched_topic():
    """Test 2: Query 'how to configure model with ConfigDict' with wrong topic='fastapi_lifespan' triggers fallback and returns pydantic_config."""
    manager = await create_test_storage_manager()
    try:
        query = "how to configure model with ConfigDict"
        # Deliberately wrong topic: fastapi_lifespan
        results = await manager.search(
            query=query,
            topic="fastapi_lifespan",
            limit=3,
        )

        assert len(results) > 0
        assert results[0].fallback_triggered is True
        assert all(r.fallback_triggered is True for r in results)
        assert results[0].chunk.topic == "pydantic_config"
        assert results[0].chunk.chunk_id == "pydantic_v2_migration#model-configuration"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_global_search_without_topic():
    """Test 3: Query with topic=None runs global search and returns relevant chunks with fallback_triggered == False."""
    manager = await create_test_storage_manager()
    try:
        query = "replace @validator with modern field validator"
        results = await manager.search(
            query=query,
            topic=None,
            limit=3,
        )

        assert len(results) > 0
        assert all(not r.fallback_triggered for r in results)
        assert results[0].chunk.topic == "pydantic_validators"
        assert results[0].chunk.chunk_id == "pydantic_v2_migration#field-validation"
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_async_context_manager_lifecycle():
    """Test 4: Tests using async with QdrantStorageManager(...) as mgr: runs search and exits cleanly."""
    client = AsyncQdrantClient(":memory:")
    embedder = CachedEmbedder(cache_path=CACHE_FILE_PATH)
    await ingest_corpus(
        data_dir=DATA_DIR,
        client=client,
        embedder=embedder,
        collection_name=DEFAULT_COLLECTION_NAME,
        recreate=True,
    )

    async with QdrantStorageManager(client=client, embedder=embedder) as mgr:
        results = await mgr.search(query="how to configure model with ConfigDict", limit=2)
        assert len(results) > 0
        assert results[0].chunk.topic == "pydantic_config"
