# Pydantic v2 Migration Copilot: Project Architecture & System Truth

## 1. Project Overview & Scope
The **Pydantic v2 Migration Copilot** is an enterprise AI pair programmer designed to modernize legacy Python codebases (Pydantic v1.x and FastAPI legacy patterns) to strict Pydantic v2 and modern FastAPI standards.

It combines:
1. **Hierarchical Multi-Agent Supervision**: CEO (User) ➔ Tech Lead (Main Agent) ➔ Specialized Worker Subagents.
2. **High-Precision Scoped RAG**: Vector search powered by local embedded Qdrant and Google GenAI `text-embedding-004`.
3. **Deterministic AST Validation**: Static code analysis enforcing zero legacy tokens before returning solutions.
4. **FastAPI Modern Backend**: Async Server-Sent Events (SSE) streaming API with non-blocking lifecycle management.

---

## 2. Locked RAG Pipeline Invariants

| Component | Technical Specification | Rationale & Failure Mode Prevented |
| :--- | :--- | :--- |
| **Corpus Sourcing** | Curated local snapshots in `src/storage/data/` | 100% hermetic offline reproducibility; prevents scraping flakiness & breaking changes. |
| **File Partitioning** | 3 topic files (`pydantic_v2_migration.md`, `pydantic_settings.md`, `fastapi_modern.md`) | Clean library boundaries; prevents micro-file fragmentation and giant monolithic smearing. |
| **Text Decoupling** | Programmatic separation of `search_text` vs `payload_content` | Prevents the **Negation Trap** where legacy code in raw markdown distorts vector centroids. |
| **Chunking Strategy** | Hierarchical header splitting (`##` & `###`) with atomic code fences | Treats ```python blocks as indivisible units; prevents `IndentationError` and broken syntax. |
| **Overlap Strategy** | Zero sliding token overlap + Parent Breadcrumb Prepending | Eliminates top-k duplicate code bloat; provides explicit context via `[File > Section > SubSection]`. |
| **Distance Metric** | `models.Distance.COSINE` (768 dimensions) | Magnitude-invariant angular comparison; avoids Euclidean distance concentration. |
| **Embedder Protocol** | `EmbedderProtocol` (Production `GeminiEmbedder` + `DeterministicMockEmbedder`) | Guarantees instant 5ms local tests without API keys, network access, or burning API quota. |
| **Storage Engine** | Qdrant Dual-Mode (Disk `./.qdrant_storage` + RAM `:memory:`) | Native async-first Rust engine; solves RocksDB multi-process lock collision during testing. |
| **Retrieval Strategy**| Scoped Payload Filtering with Graceful Fallback (`score < 0.55`) | High-precision topical filtering with automatic fallback to broad search on vocabulary drift. |

---

## 3. Storage Schema & Data Contracts

### 3.1 Qdrant Collection: `python_tooling_docs`
* **Vector Configuration**: `VectorParams(size=768, distance=Distance.COSINE)`
* **Payload Structure**:
  ```python
  {
      "chunk_id": "pydantic_v2_migration#field-validators",
      "topic": "pydantic_validators",         # Keyword index
      "source_file": "pydantic_v2_migration.md",
      "breadcrumb": "Pydantic V2 Migration > Validators > Custom Field Validation",
      "target_version": "2.8+",
      "content": "Full markdown text with complete Before/After code blocks"
  }
  ```

### 3.2 Dual Embedder Interface (`EmbedderProtocol`)
```python
import typing

@typing.runtime_checkable
class EmbedderProtocol(typing.Protocol):
    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of text strings into 768-dimensional float vectors."""
        ...

    async def embed_query(self, query: str) -> list[float]:
        """Embed a single search query string."""
        ...
```

---

## 4. Component Layout

```
pydantic-v2-copilot/
├── PROJECT.md                     # This document (Static System Truth, 0 passive tokens)
├── pyproject.toml                 # Single-TOML packaging & dependencies (uv managed)
├── src/
│   ├── storage/                   # Data Ingestion & Qdrant Vector Engine
│   │   ├── data/                  # Bundled markdown documentation snapshots
│   │   ├── chunker.py             # MarkdownHeaderChunker (AST & header splitting)
│   │   ├── embedder.py            # GeminiEmbedder & DeterministicMockEmbedder
│   │   └── qdrant_manager.py      # AsyncQdrantClient lifecycle & scoped search
│   ├── core_agents/               # Multi-Agent Orchestration & AST Validation
│   │   ├── schemas.py             # Pydantic v2 data contracts between agents
│   │   ├── decomposer.py          # Query Decomposer Agent
│   │   ├── retriever.py           # Scoped Retriever Agent
│   │   ├── synthesizer.py         # Code Modernizer Agent
│   │   └── critic.py              # AST & Rule Critic Agent (Zero v1 enforcement)
│   └── copilot_api/               # FastAPI Application & Streaming Endpoints
│       ├── main.py                # App entrypoint, lifespan event, health check
│       └── routes.py              # /v1/chat, /v1/migrate, SSE streaming
└── tests/                         # Hermetic test suite (Pytest + :memory: Qdrant)
```

---

## 5. Development & Testing Invariants
1. **Toolchain**: Strictly Astral `uv` (`uv run`, `uv sync`, `uv add`). Never bare `pip`.
2. **Pydantic v2**: Use `model_config = ConfigDict(...)`, `@field_validator`, and `.model_dump()`.
3. **Async First**: All FastAPI handlers and Qdrant queries must use `await` and `AsyncQdrantClient`.
4. **Hermetic Testing**: Tests must pass with zero network access and no `GEMINI_API_KEY` set.
