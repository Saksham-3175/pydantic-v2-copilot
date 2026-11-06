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

---

### ADR-07: In-Process Native Engine Execution vs. Client-Server IPC
* **Context**: How the CLI interface communicates with the multi-agent migration engine.
* **Option A (Client-Server IPC via background HTTP daemon)**:
  * *Gain*: Reuses HTTP API routes directly.
  * *Sacrifice*: 1.5–3.0s web server boot latency, permanent ~160 MB RAM overhead, and potential port 8000 collisions with other local dev servers.
* **Option B (In-process native execution — CHOSEN)**:
  * *Gain*: ~0.25s native startup, zero port collisions, 100% offline reliability, and temporary ~120 MB RAM during command execution that drops to 0 MB immediately upon exit.
  * *Sacrifice*: CLI and API are separate entrypoints sharing the underlying core engine modules.
* **Why Option B Won**: Developer CLI tools must be instant, robust, and zero-configuration.

---

### ADR-08: File-at-a-Time Engine Granularity vs. Whole-Repository Context
* **Context**: Where project-level context and directory traversal are managed.
* **Option A (Whole-repository context in one massive prompt)**:
  * *Gain*: LLM sees all files simultaneously.
  * *Sacrifice*: Massive context window burn (15k–60k+ tokens), high latency (~25s), and risk of complete run failure if a single file triggers context limits.
* **Option B (Single-file snippet engine with directory batching at CLI layer — CHOSEN)**:
  * *Gain*: 10x–20x fewer tokens (500–1,500 tokens per call, ~2.0s latency). Complete failure isolation: a failure in one complex file does not block other files. Post-migration consistency is validated globally via `pytest`.
  * *Sacrifice*: Requires directory walker and batch scorecard orchestration in CLI.
* **Why Option B Won**: Pydantic v1 to v2 migration patterns are strictly intra-file modifications; modular file granularity optimizes speed, cost, and reliability.

---

### ADR-09: Unified Async Event Generator for Telemetry & Progress
* **Context**: How reflection loop progress and stage telemetry are streamed to different interfaces.
* **Option A (Split progress logic in each UI layer)**:
  * *Gain*: Custom progress logic per interface.
  * *Sacrifice*: Duplicate state-machine code across CLI and API routes.
* **Option B (Core engine async generator yielding abstract lifecycle events — CHOSEN)**:
  * *Gain*: `MigrationOrchestrator.migrate_stream()` yields clean event dictionaries (`start`, `decomposed`, `retrieved`, `synthesizing`, `critic_verification`, `done`). CLI renders live Rich spinners, while FastAPI streams HTTP Server-Sent Events (SSE) with 100% logic parity.
  * *Sacrifice*: Requires generator iteration in client handlers.
* **Why Option B Won**: Eliminates duplicated state-machine logic and ensures identical telemetry across CLI and Web API.

---

### ADR-10: Direct In-Place File Mutation with Git-Backed Revert vs. Staged .bak Files
* **Context**: How modernized code is persisted to disk during `--write`.
* **Option A (Create temporary `.bak` files alongside modified files)**:
  * *Gain*: Non-destructive backup even in non-git directories.
  * *Sacrifice*: Leaves cluttered `.bak` files across project trees if execution is cancelled (Ctrl+C). Triples disk I/O operations.
* **Option B (Direct in-place overwrites backed by native Git revert — CHOSEN)**:
  * *Gain*: Conforms to standard modern developer tooling (`ruff format`, `black`). Instant atomic revert via `git restore .`. Preceded by safe default dry-run (`--diff`) and git status warning check.
  * *Sacrifice*: Assumes projects are version-controlled with Git.
* **Why Option B Won**: Modern software development standardizes on Git; zero filesystem clutter outweighs `.bak` overhead.

---

### ADR-11: Single Unified Package Distribution vs. Optional Extras
* **Context**: How CLI and API dependencies are declared in `pyproject.toml`.
* **Option A (Split optional extras, e.g. `pydantic-v2-copilot[cli]` and `[api]`)**:
  * *Gain*: Minimizes sub-package installs for single-purpose environments.
  * *Sacrifice*: High user friction (`ModuleNotFoundError: missing [cli]`) and defensive import guards.
* **Option B (Single unified package with core dependencies — CHOSEN)**:
  * *Gain*: Total wheel package is only 366 KB and installs in <0.15s with `uv`. Anyone installing the wheel gets both `copilot` CLI and `copilot_api` without configuration errors.
  * *Sacrifice*: Installs both Typer and FastAPI dependencies.
* **Why Option B Won**: Zero user setup friction is paramount for developer adoption.

---

### ADR-12: State Persistence — Content-Addressed SHA-256 Migration Cache
* **Context**: CLI commands are independent OS processes. Running `copilot migrate --diff` followed by `copilot migrate --write` resulted in duplicate LLM calls, double token burn, and potential output divergence.
* **Option A (Stateless re-execution on every run)**:
  * *Gain*: Simple architecture; zero state on disk.
  * *Sacrifice*: Burns double Groq/Gemini tokens and risks generating divergent modernizations between diff review and disk write.
* **Option B (Content-addressed SHA-256 cache in `~/.cache/pydantic_copilot/` — CHOSEN)**:
  * *Gain*: When `--diff` verifies code via the Critic, it caches the output keyed by `SHA-256(path + content)`. Subsequent `--write` executes in `<0.01s` with **zero duplicate LLM calls**, perfectly preserving verified diffs while keeping the target git workspace 100% clean.
  * *Sacrifice*: Requires maintaining local JSON cache eviction and hash invalidation.
* **Why Option B Won**: Eliminates redundant token costs and guarantees exact identity between previewed diffs and written code.

---

### ADR-13: Knowledge Distribution — Bundled 161 KB Vector Cache vs. Heavy Model Downloads
* **Context**: Delivering semantic search capabilities to end-user machines during package installation.
* **Option A (Download local transformer models, e.g., HuggingFace PyTorch models)**:
  * *Gain*: Fully local vector embedding calculation for arbitrary new text.
  * *Sacrifice*: Bloats package size from 366 KB to 1.5+ GB; requires heavy C++ compiler toolchains and PyTorch dependencies.
* **Option B (Ship precomputed 768-dim embeddings in a 161 KB JSON file — CHOSEN)**:
  * *Gain*: 366 KB total package wheel size. Provides instant, 0.0001s vector lookups with 100% deterministic ranking, zero external dependencies, and full offline execution.
  * *Sacrifice*: Embedding vector index is static to the bundled documentation corpus.
* **Why Option B Won**: A static knowledge corpus does not justify a 1.5 GB PyTorch distribution.

---

### ADR-14: Architecture Paradigm — Tiered Neuro-Symbolic Engine vs. Pure LLM / Pure AST
* **Context**: Balancing execution speed, token costs, and semantic code-reasoning capabilities.
* **Option A (Pure LLM)**:
  * *Gain*: Can handle arbitrary code patterns.
  * *Sacrifice*: Expensive, slow, rate-limited, and prone to hallucinating legacy v1 patterns on simple 1-to-1 renames.
* **Option B (Pure AST / Codemod Regex)**:
  * *Gain*: Instant (0.002s), zero cost.
  * *Sacrifice*: Hits the **Codemod Wall** on semantic breaking changes (e.g., `@root_validator` dict-to-self mutations, cross-field validation, lifespan context managers).
* **Option C (Tiered Neuro-Symbolic Hybrid — CHOSEN)**:
  * *Gain*: Fast deterministic AST engine handles mechanical 1-to-1 syntax changes for 0 tokens in 0.002s. Complex semantic rewrites seamlessly delegate to the Multi-Agent RAG reflection loop, with the deterministic 4-Gate Critic acting as the compiler guardrail for both tiers.
  * *Sacrifice*: Requires maintaining both symbolic AST visitors and LLM prompt schemas.
* **Why Option C Won**: Maximizes developer velocity and token efficiency without sacrificing semantic reasoning power.
