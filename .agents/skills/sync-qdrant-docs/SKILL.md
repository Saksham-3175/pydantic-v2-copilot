---
name: sync-qdrant-docs
description: >-
  Ingest, chunk, embed, and synchronize local markdown documentation into the Qdrant vector
  database collection 'python_tooling_docs'. Use this skill when initializing or updating
  the RAG knowledge corpus, re-indexing migration documents, or checking Qdrant collection health.
---

# Sync Qdrant Documentation Knowledge Base

This skill guides the agent through chunking, embedding, and synchronizing local Pydantic v2 and FastAPI markdown documents into the local Qdrant vector database.

## System Invariants & Schemas
Before synchronizing, ensure you follow the locked system invariants defined in [PROJECT.md](../../../pydantic-v2-copilot/PROJECT.md):
* **Collection Name**: `python_tooling_docs`
* **Vector Configuration**: 768 dimensions, `Distance.COSINE`
* **Chunking**: Header-based hierarchical splitting on `##` and `###` with atomic code blocks (` ```python `)
* **Decoupled Search**: Vectors are generated from clean `search_text`; full markdown is stored in `payload["content"]`

---

## Step-by-Step Synchronization Procedure

### Step 1: Verify Corpus Source Files
Confirm that the canonical documentation snapshots exist in `pydantic-v2-copilot/src/storage/data/`:
1. `pydantic_v2_migration.md` (Validators, ConfigDict, root_validators, serialization)
2. `pydantic_settings.md` (BaseSettings separation, SettingsConfigDict)
3. `fastapi_modern.md` (Lifespan handlers, Annotated Depends)

### Step 2: Execute Ingestion Pipeline
Run the ingestion script within the project environment using Astral `uv`:

```bash
cd /home/gg31/Work/agy-code/pydantic-v2-copilot
uv run python -m storage.ingest
```

> [!NOTE]
> If `GEMINI_API_KEY` is not present in the environment, the ingestion runner should automatically fall back to `DeterministicMockEmbedder` to allow hermetic local testing without crashing.

### Step 3: Validate Collection Health & Point Counts
Verify that vectors were correctly indexed by executing a quick health check:

```bash
uv run python -c "
import asyncio
from qdrant_client import AsyncQdrantClient

async def check():
    client = AsyncQdrantClient(path='./.qdrant_storage')
    info = await client.get_collection('python_tooling_docs')
    print(f'Collection Status: {info.status}, Points Count: {info.points_count}')
    assert info.points_count > 0, 'No points indexed!'

asyncio.run(check())
"
```

### Step 4: Verification of Scoped Retrieval
Test that a sample scoped vector search succeeds:
* Query: `"how to replace @validator with @field_validator"`
* Topic Filter: `pydantic_validators`
* Expected outcome: Returns chunk containing `@field_validator` with cosine similarity `> 0.70`.
