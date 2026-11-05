# Architecture Decision Records (ADRs)

Engineering decisions, considered alternatives, and explicit trade-offs for the **Pydantic v2 Migration Copilot**.

---

### ADR-01: MVP Scope — Minimal Spanning Basis vs. Full Corpus Ingestion
* **Context**: Pydantic v2 introduced dozens of minor changes across its API.
* **Option A (Ingest entire 50,000-word documentation corpus)**:
  * *Gain*: Broad vanity coverage.
  * *Sacrifice*: High token burn, slow test cycles, and high risk of subtle syntax hallucinations across unverified edge cases.
* **Option B (Curate the 6-change Minimal Spanning Basis — CHOSEN)**:
  * *Gain*: 100% mathematical coverage of all orthogonal breaking syntax (ConfigDict, field_validator, model_validator, serialization, BaseSettings, FastAPI lifespan) with sub-second hermetic test loops.
  * *Sacrifice*: Defers rare edge cases (e.g., custom root types, generic models) to Phase 6 enterprise expansion.
* **Why Option B Won**: Proving deterministic end-to-end correctness and AST zero-regression gates on the 6 basis vectors is necessary before scaling corpus volume.

---

### ADR-02: Storage Engine — Qdrant vs. ChromaDB
* **Context**: The RAG pipeline requires local dense vector storage and metadata filtering.
* **Option A (ChromaDB)**:
  * *Gain*: Popular in basic tutorials; minimal initial setup.
  * *Sacrifice*: Synchronous architecture blocks FastAPI's async event loop. Embedded SQLite/RocksDB engine acquires an exclusive OS file lock, causing `RuntimeError: Storage folder is already accessed by another instance` whenever tests and the live server run concurrently.
* **Option B (Qdrant Dual-Mode — CHOSEN)**:
  * *Gain*: Native async Rust client (`AsyncQdrantClient`) avoids event loop starvation. Seamless dual-mode: persistent on-disk storage (`./.qdrant_storage`) for the API server, and ephemeral in-memory storage (`:memory:`) for 5ms isolated test runs with zero file-lock collisions.
  * *Sacrifice*: Requires explicit vector parameter schemas and payload indexing.
* **Why Option B Won**: Eliminates file-lock collisions during concurrent test execution and provides non-blocking async throughput for FastAPI.

---

### ADR-03: Corpus Sourcing — Bundled Markdown Snapshots vs. Dynamic Web Scraping
* **Context**: The RAG pipeline needs authoritative Pydantic v2 and FastAPI documentation.
* **Option A (Dynamic runtime web scraping of documentation sites)**:
  * *Gain*: Automatically tracks website changes.
  * *Sacrifice*: Fragile (breaks on layout redesigns or network drops), introduces 2–3s network latency, and pollutes embeddings with HTML boilerplate (cookie banners, sidebars, navigation links).
* **Option B (Bundled local Markdown snapshots from upstream GitHub — CHOSEN)**:
  * *Gain*: 100% hermetic offline reproducibility, zero network dependency in CI/CD, sub-second ingestion, and clean text payloads.
  * *Sacrifice*: Requires manual re-syncing when major upstream versions release.
* **Why Option B Won**: Embedding purity and offline test reproducibility outweigh live web scraping.

---

### ADR-04: Chunking Strategy — Custom State-Machine Chunker vs. Framework Splitters (LangChain)
* **Context**: Technical documentation mixes explanatory prose with Python code blocks.
* **Option A (Framework sliding character/token splitters, e.g., LangChain `RecursiveCharacterTextSplitter`)**:
  * *Gain*: Single library import; zero custom parsing code.
  * *Sacrifice*: Arbitrarily splits text every 500 characters, severing Python classes and functions mid-indentation and causing fatal `IndentationError` when retrieved into LLM prompts. Pulls 15+ heavy external dependencies.
* **Option B (Custom `MarkdownHeaderChunker` — CHOSEN)**:
  * *Gain*: ~75 lines of standard library Python (`re`, `dataclasses`). Slices strictly on Markdown headers (`##`, `###`) and enforces the **Atomic Code Rule** (fenced code blocks are indivisible units that are never split).
  * *Sacrifice*: Requires maintaining custom parsing logic.
* **Why Option B Won**: Code syntax integrity is non-negotiable in an automated code migration engine.

---

### ADR-05: Embedding Representation — Decoupled Storage vs. Raw Markdown Embeddings
* **Context**: Migration documentation contrasts deprecated v1 patterns directly alongside modern v2 patterns.
* **Option A (Embed raw markdown directly)**:
  * *Gain*: Simple 1:1 mapping (one string per chunk).
  * *Sacrifice*: Triggers the **Negation Trap**: sentences like *"WARNING: Do NOT use `@validator`"* poison vector centroids with legacy tokens, causing searches for legacy code to mistakenly retrieve negative warnings.
* **Option B (Decoupled representation: `search_text` vs. `payload_content` — CHOSEN)**:
  * *Gain*: `search_text` strips out legacy blocks and negative warnings, embedding ONLY target modern v2 signatures (clean vector centroid). `payload_content` preserves full Before/After context for the LLM.
  * *Sacrifice*: Requires a two-pass extraction step during chunking.
* **Why Option B Won**: Prevents deprecation warnings from corrupting vector similarity scores while ensuring the LLM receives complete explanatory context.

---

### ADR-06: Retrieval Strategy — Automatic Fallback Guardrail vs. Strict Metadata Filtering
* **Context**: Retrieval precision improves when filtering vectors by topic (`pydantic_validators`, `pydantic_config`, etc.).
* **Option A (Strict topic payload filtering)**:
  * *Gain*: High precision within known categories.
  * *Sacrifice*: The **False-Negative Trap**: if the Decomposer agent misclassifies a query due to vocabulary drift, the search returns 0 results and starves the Synthesizer of context.
* **Option B (Filtered search with Automatic Fallback Guardrail at `score < 0.55` — CHOSEN)**:
  * *Gain*: Searches topic aisles first. If top cosine similarity is `< 0.55` (doubtful match), the manager automatically drops the topic filter and runs a global dense search.
  * *Sacrifice*: Adds a second search call (~5ms) when fallback triggers.
* **Why Option B Won**: Preventing false-negative silence is worth the negligible ~5ms fallback search cost.
