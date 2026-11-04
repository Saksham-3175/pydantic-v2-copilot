from __future__ import annotations

from contextlib import asynccontextmanager
import os
from pathlib import Path
import typing

from fastapi import FastAPI
from qdrant_client import AsyncQdrantClient

from copilot_api.routes import router
from core_agents.orchestrator import MigrationOrchestrator
from core_agents.retriever import ScopedRetriever
from storage.embedder import CachedEmbedder, get_default_embedder
from storage.ingest import DEFAULT_COLLECTION_NAME, ensure_collection, ingest_corpus
from storage.qdrant_manager import QdrantStorageManager


@asynccontextmanager
async def lifespan(app: FastAPI) -> typing.AsyncGenerator[None, None]:
    """Modern FastAPI lifespan managing AsyncQdrantClient and MigrationOrchestrator lifecycle."""
    data_dir = Path(__file__).resolve().parent.parent / "storage" / "data"
    cache_path = data_dir / "gemini_embeddings_cache.json"

    # Embedder with cache for 100% offline hermetic execution
    if cache_path.exists():
        embedder = CachedEmbedder(cache_path=cache_path)
    else:
        embedder = get_default_embedder()

    storage_path = Path("./.qdrant_storage")
    is_test = bool(os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("TESTING"))

    client: AsyncQdrantClient
    if not is_test and storage_path.exists():
        try:
            client = AsyncQdrantClient(path=str(storage_path))
            await ensure_collection(client, collection_name=DEFAULT_COLLECTION_NAME)
        except Exception:
            client = AsyncQdrantClient(":memory:")
            await ensure_collection(client, collection_name=DEFAULT_COLLECTION_NAME)
            if data_dir.exists():
                await ingest_corpus(data_dir=data_dir, client=client, embedder=embedder)
    else:
        client = AsyncQdrantClient(":memory:")
        await ensure_collection(client, collection_name=DEFAULT_COLLECTION_NAME)
        if data_dir.exists():
            await ingest_corpus(data_dir=data_dir, client=client, embedder=embedder)

    storage_manager = QdrantStorageManager(
        client=client,
        collection_name=DEFAULT_COLLECTION_NAME,
        embedder=embedder,
    )
    retriever = ScopedRetriever(storage_manager=storage_manager)
    orchestrator = MigrationOrchestrator(retriever=retriever)

    app.state.qdrant_client = client
    app.state.storage_manager = storage_manager
    app.state.orchestrator = orchestrator

    try:
        yield
    finally:
        await storage_manager.close()


app = FastAPI(
    title="Pydantic v2 Copilot API",
    version="0.1.0",
    description="Multi-Agent RAG Copilot for Modern Python & Pydantic v2 Migration",
    lifespan=lifespan,
)

# Mount routes with /v1 prefix and root prefix
app.include_router(router, prefix="/v1", tags=["Migration"])
app.include_router(router, tags=["Migration"])


@app.get("/health")
async def health_check():
    """Health check endpoint confirming API status."""
    return {
        "status": "healthy",
        "service": "pydantic-v2-copilot",
        "version": "0.1.0",
    }
