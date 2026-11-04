"""Integration and unit tests for Qdrant corpus ingestion runner."""

from pathlib import Path
import tempfile
import uuid
import pytest
from qdrant_client import AsyncQdrantClient, models

from storage.chunker import MarkdownHeaderChunker
from storage.embedder import DeterministicMockEmbedder
from storage.ingest import (
    DEFAULT_COLLECTION_NAME,
    ensure_collection,
    ingest_corpus,
    main,
    parse_args,
)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "src" / "storage" / "data"


@pytest.mark.asyncio
async def test_ensure_collection():
    """Verify collection creation with 768-dim Cosine distance in :memory:."""
    client = AsyncQdrantClient(":memory:")
    collection_name = "test_ensure"

    # Verify collection does not exist initially
    assert not await client.collection_exists(collection_name)

    # Ensure collection creates it with 768-dim Cosine
    await ensure_collection(client, collection_name=collection_name)
    assert await client.collection_exists(collection_name)

    info = await client.get_collection(collection_name)
    assert info.config.params.vectors.size == 768
    assert info.config.params.vectors.distance == models.Distance.COSINE

    # Calling ensure_collection again with recreate=False should be idempotent
    await ensure_collection(client, collection_name=collection_name, recreate=False)
    assert await client.collection_exists(collection_name)

    await client.close()


@pytest.mark.asyncio
async def test_ingest_corpus_point_count():
    """Run ingest_corpus on CORPUS_DIR into :memory:, asserting return value == 8 and info.points_count == 8."""
    client = AsyncQdrantClient(":memory:")
    embedder = DeterministicMockEmbedder()

    count = await ingest_corpus(
        data_dir=CORPUS_DIR,
        client=client,
        embedder=embedder,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    assert count == 8
    info = await client.get_collection(DEFAULT_COLLECTION_NAME)
    assert info.points_count == 8

    await client.close()


@pytest.mark.asyncio
async def test_ingest_payload_integrity():
    """Verify payload integrity for pydantic_v2_migration#field-validation chunk."""
    client = AsyncQdrantClient(":memory:")
    embedder = DeterministicMockEmbedder()

    await ingest_corpus(
        data_dir=CORPUS_DIR,
        client=client,
        embedder=embedder,
        collection_name=DEFAULT_COLLECTION_NAME,
    )

    point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, "pydantic_v2_migration#field-validation"))
    records = await client.retrieve(
        collection_name=DEFAULT_COLLECTION_NAME,
        ids=[point_id],
    )

    assert len(records) == 1
    point = records[0]
    payload = point.payload
    assert payload is not None
    assert payload["topic"] == "pydantic_validators"
    assert "breadcrumb" in payload
    assert "content" in payload
    assert "Field Validation" in payload["breadcrumb"]
    assert "field_validator" in payload["content"]

    await client.close()


@pytest.mark.asyncio
async def test_recreate_collection():
    """Test that running with recreate=True resets and re-populates points without duplicate errors."""
    client = AsyncQdrantClient(":memory:")
    embedder = DeterministicMockEmbedder()

    # Initial ingestion
    count1 = await ingest_corpus(
        data_dir=CORPUS_DIR,
        client=client,
        embedder=embedder,
        collection_name=DEFAULT_COLLECTION_NAME,
        recreate=False,
    )
    assert count1 == 8
    info1 = await client.get_collection(DEFAULT_COLLECTION_NAME)
    assert info1.points_count == 8

    # Recreate collection and re-ingest
    count2 = await ingest_corpus(
        data_dir=CORPUS_DIR,
        client=client,
        embedder=embedder,
        collection_name=DEFAULT_COLLECTION_NAME,
        recreate=True,
    )
    assert count2 == 8
    info2 = await client.get_collection(DEFAULT_COLLECTION_NAME)
    assert info2.points_count == 8

    # Idempotent re-run with recreate=False preserves count and updates points
    count3 = await ingest_corpus(
        data_dir=CORPUS_DIR,
        client=client,
        embedder=embedder,
        collection_name=DEFAULT_COLLECTION_NAME,
        recreate=False,
    )
    assert count3 == 8
    info3 = await client.get_collection(DEFAULT_COLLECTION_NAME)
    assert info3.points_count == 8

    await client.close()


@pytest.mark.asyncio
async def test_ingest_empty_directory():
    """Ingesting an empty directory should return 0 and create the collection."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        client = AsyncQdrantClient(":memory:")
        count = await ingest_corpus(
            data_dir=Path(tmp_dir),
            client=client,
            embedder=DeterministicMockEmbedder(),
            collection_name="empty_test",
        )
        assert count == 0
        assert await client.collection_exists("empty_test")
        info = await client.get_collection("empty_test")
        assert info.points_count == 0
        await client.close()


def test_parse_args_defaults():
    """Verify CLI parser default arguments."""
    args = parse_args([])
    assert args.data_dir == Path("src/storage/data")
    assert args.storage_path == "./.qdrant_storage"
    assert args.cache_file == Path("src/storage/data/gemini_embeddings_cache.json")
    assert not args.in_memory
    assert not args.recreate
    assert not args.dry_run
    assert not args.mock


def test_parse_args_custom():
    """Verify CLI parser with custom arguments."""
    args = parse_args([
        "--data-dir", "custom/data",
        "--storage-path", "/tmp/qdrant",
        "--cache-file", "/tmp/custom_cache.json",
        "--in-memory",
        "--recreate",
        "--dry-run",
        "--mock",
    ])
    assert args.data_dir == Path("custom/data")
    assert args.storage_path == "/tmp/qdrant"
    assert args.cache_file == Path("/tmp/custom_cache.json")
    assert args.in_memory
    assert args.recreate
    assert args.dry_run
    assert args.mock


@pytest.mark.asyncio
async def test_cli_main_dry_run(capsys: pytest.CaptureFixture[str]):
    """Verify CLI runner main() with dry-run and mock flags."""
    await main(["--dry-run", "--mock", "--data-dir", str(CORPUS_DIR)])
    captured = capsys.readouterr()
    assert "[DRY-RUN] Successfully chunked 8 documents" in captured.out
    assert "[DRY-RUN] Successfully generated 8 embeddings" in captured.out


@pytest.mark.asyncio
async def test_cli_main_in_memory(capsys: pytest.CaptureFixture[str]):
    """Verify CLI runner main() with in-memory execution and mock embedder."""
    await main(["--in-memory", "--mock", "--data-dir", str(CORPUS_DIR)])
    captured = capsys.readouterr()
    assert "Successfully ingested 8 points into collection 'python_tooling_docs'" in captured.out
