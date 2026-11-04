"""Integration tests verifying Gemini embeddings cache integrity and real semantic retrieval."""

import json
import math
from pathlib import Path
import pytest
from qdrant_client import AsyncQdrantClient

from storage.embedder import CachedEmbedder, DeterministicMockEmbedder
from storage.ingest import DEFAULT_COLLECTION_NAME, ingest_corpus

CACHE_FILE_PATH = (
    Path(__file__).resolve().parent.parent
    / "src"
    / "storage"
    / "data"
    / "gemini_embeddings_cache.json"
)

DATA_DIR = Path(__file__).resolve().parent.parent / "src" / "storage" / "data"

EXPECTED_CHUNK_IDS = {
    "fastapi_modern#application-lifespan",
    "fastapi_modern#dependency-injection",
    "pydantic_settings#basesettings-migration",
    "pydantic_settings#environment-parsing-precedence",
    "pydantic_v2_migration#field-validation",
    "pydantic_v2_migration#model-configuration",
    "pydantic_v2_migration#root-and-model-validators",
    "pydantic_v2_migration#serialization",
}


def test_gemini_cache_file_integrity():
    """Verify gemini_embeddings_cache.json contains all 8 chunk IDs and each vector has length 768 with unit L2 norm."""
    assert CACHE_FILE_PATH.exists(), f"Embeddings cache file not found at {CACHE_FILE_PATH}"

    with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    assert isinstance(cache, dict)
    assert len(cache) == 8, f"Expected 8 cached chunks, found {len(cache)}"
    assert set(cache.keys()) == EXPECTED_CHUNK_IDS

    for chunk_id, vector in cache.items():
        assert isinstance(vector, list), f"Vector for {chunk_id} must be a list"
        assert len(vector) == 768, f"Vector for {chunk_id} must have length 768, got {len(vector)}"
        assert all(isinstance(val, float) for val in vector), f"All elements in {chunk_id} vector must be floats"

        # Verify unit L2 norm (1.0 +/- 1e-4)
        l2_norm = math.sqrt(sum(x * x for x in vector))
        assert math.isclose(l2_norm, 1.0, rel_tol=1e-4), (
            f"Vector for {chunk_id} has L2 norm {l2_norm}, expected 1.0 +/- 1e-4"
        )


@pytest.mark.asyncio
async def test_real_cached_semantic_search():
    """Load the 8 real cached Gemini vectors into AsyncQdrantClient(':memory:') and assert semantic search precision."""
    client = AsyncQdrantClient(":memory:")
    embedder = CachedEmbedder(cache_path=CACHE_FILE_PATH)

    try:
        # Ingest corpus into in-memory Qdrant using the CachedEmbedder
        count = await ingest_corpus(
            data_dir=DATA_DIR,
            client=client,
            embedder=embedder,
            collection_name=DEFAULT_COLLECTION_NAME,
            recreate=True,
        )
        assert count == 8

        query = "replace @validator with modern field validator"
        query_vec = await embedder.embed_query(query)
        assert len(query_vec) == 768

        # Execute semantic retrieval query against in-memory collection
        search_result = await client.query_points(
            collection_name=DEFAULT_COLLECTION_NAME,
            query=query_vec,
            limit=3,
        )

        assert len(search_result.points) > 0, "Semantic search returned no results"

        top_hit = search_result.points[0]
        assert top_hit.payload is not None
        top_chunk_id = top_hit.payload.get("chunk_id")

        # Assert top retrieved point is pydantic_v2_migration#field-validation
        assert top_chunk_id == "pydantic_v2_migration#field-validation", (
            f"Expected top hit 'pydantic_v2_migration#field-validation', got '{top_chunk_id}'"
        )

        # Assert cosine similarity score is > 0.65
        assert top_hit.score > 0.65, (
            f"Cosine similarity score {top_hit.score:.4f} is not > 0.65"
        )
    finally:
        await client.close()


@pytest.mark.asyncio
async def test_cached_embedder_offline_fallback():
    """Verify CachedEmbedder gracefully falls back to DeterministicMockEmbedder when keys are missing."""
    embedder = CachedEmbedder(
        cache_path=CACHE_FILE_PATH,
        fallback_embedder=DeterministicMockEmbedder(),
    )

    # Embedding an unknown text returns a valid 768-dim vector from fallback
    unknown_vecs = await embedder.embed_texts(["completely unknown text phrase"])
    assert len(unknown_vecs) == 1
    assert len(unknown_vecs[0]) == 768
    norm = math.sqrt(sum(x * x for x in unknown_vecs[0]))
    assert math.isclose(norm, 1.0, rel_tol=1e-4)

    # Known chunk_id returns precomputed vector matching cache exactly
    with open(CACHE_FILE_PATH, "r", encoding="utf-8") as f:
        cache = json.load(f)

    cached_vecs = await embedder.embed_texts(["pydantic_v2_migration#field-validation"])
    assert cached_vecs[0] == cache["pydantic_v2_migration#field-validation"]
