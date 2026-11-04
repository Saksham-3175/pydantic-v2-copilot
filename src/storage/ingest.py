"""Asynchronous corpus ingestion runner and collection management for Qdrant vector database."""

import argparse
import asyncio
import json
from pathlib import Path
import sys
import typing
import uuid

from qdrant_client import AsyncQdrantClient, models

from storage.chunker import DocChunk, MarkdownHeaderChunker
from storage.embedder import DeterministicMockEmbedder, EmbedderProtocol, get_default_embedder

DEFAULT_COLLECTION_NAME = "python_tooling_docs"
VECTOR_DIMENSION = 768


async def ensure_collection(
    client: AsyncQdrantClient,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    recreate: bool = False,
) -> None:
    """Ensure target Qdrant collection exists with 768-dim Cosine distance."""
    exists = await client.collection_exists(collection_name)
    if recreate and exists:
        await client.delete_collection(collection_name)
        exists = False

    if not exists:
        await client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(
                size=VECTOR_DIMENSION,
                distance=models.Distance.COSINE,
            ),
        )


async def ingest_corpus(
    data_dir: Path,
    client: AsyncQdrantClient,
    embedder: typing.Optional[EmbedderProtocol] = None,
    collection_name: str = DEFAULT_COLLECTION_NAME,
    recreate: bool = False,
    cache_file: typing.Optional[typing.Union[str, Path]] = None,
) -> int:
    """Ingest markdown documentation corpus into Qdrant vector database.

    1. Calls `await ensure_collection(client, collection_name, recreate=recreate)`.
    2. Chunks markdown files in `data_dir` using `MarkdownHeaderChunker().chunk_directory(data_dir)`.
    3. If `embedder` is None, calls `get_default_embedder()`.
    4. Extracts search texts: `texts = [c.search_text for c in chunks]`.
    5. Embeds search texts: `vectors = await embedder.embed_texts(texts)`.
    6. If cache_file provided, serializes `{chunk.chunk_id: vector}` to JSON.
    7. Builds points with deterministic UUID5 IDs and payload dictionaries.
    8. Upserts points into Qdrant collection.

    Returns the total count of points ingested (e.g. 8).
    """
    await ensure_collection(client, collection_name=collection_name, recreate=recreate)

    chunker = MarkdownHeaderChunker()
    chunks: list[DocChunk] = chunker.chunk_directory(data_dir)
    if not chunks:
        return 0

    if embedder is None:
        embedder = get_default_embedder()

    texts = [c.search_text for c in chunks]
    vectors = await embedder.embed_texts(texts)

    if cache_file:
        cache_path = Path(cache_file)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_data = {chunk.chunk_id: vector for chunk, vector in zip(chunks, vectors)}
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache_data, f, indent=2)

    points: list[models.PointStruct] = []
    for chunk, vector in zip(chunks, vectors):
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk.chunk_id))
        points.append(
            models.PointStruct(
                id=point_id,
                vector=vector,
                payload=chunk.to_dict(),
            )
        )

    await client.upsert(collection_name=collection_name, points=points)
    return len(points)


def parse_args(args: typing.Optional[list[str]] = None) -> argparse.Namespace:
    """Parse CLI arguments for corpus ingestion runner."""
    parser = argparse.ArgumentParser(
        description="Ingest markdown documentation corpus into Qdrant vector collection."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("src/storage/data"),
        help="Directory containing markdown corpus (default: src/storage/data)",
    )
    parser.add_argument(
        "--storage-path",
        type=str,
        default="./.qdrant_storage",
        help="Local on-disk storage path for Qdrant (default: ./.qdrant_storage)",
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=Path("src/storage/data/gemini_embeddings_cache.json"),
        help="Path to save precomputed embeddings JSON cache (default: src/storage/data/gemini_embeddings_cache.json)",
    )
    parser.add_argument(
        "--in-memory",
        action="store_true",
        help="Use ephemeral in-memory Qdrant storage (:memory:)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Recreate (drop and recreate) the collection before ingestion",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chunk and embed without writing to disk / Qdrant",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force use of DeterministicMockEmbedder for hermetic offline execution",
    )
    return parser.parse_args(args)


async def main(args: typing.Optional[list[str]] = None) -> None:
    """Execute corpus ingestion from CLI arguments."""
    parsed_args = parse_args(args)
    data_dir = parsed_args.data_dir.resolve()

    if parsed_args.mock:
        embedder: EmbedderProtocol = DeterministicMockEmbedder()
    else:
        embedder = get_default_embedder()

    if parsed_args.dry_run:
        chunker = MarkdownHeaderChunker()
        chunks = chunker.chunk_directory(data_dir)
        texts = [c.search_text for c in chunks]
        vectors = await embedder.embed_texts(texts)
        print(f"[DRY-RUN] Successfully chunked {len(chunks)} documents from {data_dir}.")
        print(f"[DRY-RUN] Successfully generated {len(vectors)} embeddings (dim={len(vectors[0]) if vectors else 0}).")
        print("[DRY-RUN] No writes performed to Qdrant storage.")
        return

    if parsed_args.in_memory:
        client = AsyncQdrantClient(":memory:")
    else:
        client = AsyncQdrantClient(path=parsed_args.storage_path)

    # Protect precomputed real cache from being overwritten by default mock CLI runs
    if parsed_args.mock and parsed_args.cache_file == Path("src/storage/data/gemini_embeddings_cache.json"):
        cache_file = None
    else:
        cache_file = parsed_args.cache_file

    try:
        count = await ingest_corpus(
            data_dir=data_dir,
            client=client,
            embedder=embedder,
            collection_name=DEFAULT_COLLECTION_NAME,
            recreate=parsed_args.recreate,
            cache_file=cache_file,
        )
        print(f"Successfully ingested {count} points into collection '{DEFAULT_COLLECTION_NAME}'.")
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
