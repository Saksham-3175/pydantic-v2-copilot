# Architecture Decision Records (ADRs) & Engineering Trade-offs

A comprehensive engineering reference documenting the technical trade-offs, rejected alternatives, and failure modes resolved in the **Pydantic v2 Migration Copilot**.

---

## 🧭 Executive Summary: Systems Over Scripts

Most AI pair programmers are fragile wrappers: they paste user code into a prompt, call an LLM once, and return unverified code. When modernizing legacy Python codebases (Pydantic v1 ➔ v2), this naive approach fails because LLMs hallucinate legacy tokens (`@validator`, `pre=True`, inner `class Config:`).

This project was engineered as a **closed-loop, compiler-verified multi-agent system**. Every component was designed with deliberate trade-offs to eliminate silent production failures:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              THE 8 CORE ARCHITECTURAL DECISIONS                        │
├────────────────────────────────────────────────────────────────────────────────────────┤
│ ADR-01: The Negation Trap Solution       ──▶ Programmatic decoupling of vector search  │
│ ADR-02: The Indentation Guard            ──▶ Custom state-machine markdown chunker     │
│ ADR-03: Deterministic AST Critic         ──▶ Static Python AST gates (zero LLM judge)  │
│ ADR-04: Qdrant vs. ChromaDB              ──▶ Solving the RocksDB multi-process lock    │
│ ADR-05: 6-Question Minimal Spanning Basis──▶ Orthogonal linear basis vs. 600 tests     │
│ ADR-06: Dogfooding Pydantic v2           ──▶ Native structured output vs. dataclasses  │
│ ADR-07: Dual Embedder & Offline Cache    ──▶ Hermetic 5ms CI runs with real vectors    │
│ ADR-08: The Ghost Ship Disconnect Guard  ──▶ Aborting SSE agent loops on tab close     │
└────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## ADR-01: The Negation Trap Solution (Decoupled Representation)

### Context & Failure Mode
Official migration guides frequently contrast obsolete syntax with modern replacements:
> *"WARNING: Do **NOT** use `@validator` or `class Config:` in Pydantic v2."*

If raw documentation is embedded directly into vectors, the vector is heavily contaminated with the legacy tokens (`@validator`, `Config`). When a user inputs legacy code, dense similarity search matches the negative warnings, confusing the downstream LLM into reproducing legacy syntax.

### Options Considered
* **Option A (Embed Raw Markdown)**: Simple, zero custom code. **Rejected**: Vector centroid is corrupted by legacy words; top hits return deprecation warnings instead of target modern signatures.
* **Option B (Decoupled Representation - CHOSEN)**: Programmatically separate what we search on from what we pass to the LLM.

### Implementation
In `src/storage/chunker.py`:
1. **`is_legacy_code_block()`**: Detects whether a code block begins with markers like `# Legacy Pydantic v1 Pattern (Deprecated)`.
2. **`search_text`**: Strips out all legacy code blocks and negative warnings, retaining *only* modern v2 target signatures and structural breadcrumbs. Vector embeddings point strictly to the modern destination.
3. **`payload_content`**: Retains the rich Before/After examples and explanatory prose, delivered to the LLM Synthesizer once the vector match is made.

---

## ADR-02: The Indentation Guard (State-Machine Markdown Chunker)

### Context & Failure Mode
Standard RAG frameworks (LangChain, LlamaIndex) rely on character or token sliding windows (e.g., `RecursiveCharacterTextSplitter`). When applied to Python code, sliding windows indiscriminately sever classes and functions mid-indentation, causing fatal `IndentationError` when retrieved into an LLM prompt.

### Options Considered
* **Option A (LangChain / LlamaIndex Splitters)**: Quick to install, but pulls 15+ transitive dependencies (`tiktoken`, `fsspec`, `tenacity`) and chops Python code mid-block. **Rejected**.
* **Option B (Custom `MarkdownHeaderChunker` - CHOSEN)**: ~75 lines of standard library Python (`re`, `dataclasses`).

### Implementation
In `src/storage/chunker.py`:
* Partitions documents strictly along hierarchical Markdown headers (`## ` and `### `).
* Employs an internal state machine (`in_code_block: bool`). Fenced code blocks (` ```python ... ``` `) are treated as indivisible atomic units that are **never split across chunk boundaries**.
* Prepends parent breadcrumbs (`Context: [File > Section > SubSection]`), eliminating redundant overlapping windows while preserving structural context.

---

## ADR-03: Deterministic Static AST Critic vs. "LLM-as-a-Judge"

### Context & Failure Mode
AI-generated code must be verified before being returned to users. Most tutorials use an LLM prompt as an evaluator (*"Judge if this code is modern Pydantic v2"*). LLMs are polite, stochastic, and hallucinate that broken code looks plausible.

### Options Considered
* **Option A (LLM-as-a-Judge)**: Non-deterministic, costs $0.02 and 2–4 seconds of latency per check, and hallucinates that legacy arguments like `@validator(pre=True)` are valid. **Rejected**.
* **Option B (Deterministic Python AST Critic - CHOSEN)**: Native Python `ast.parse()` and `ast.NodeVisitor`.

### Implementation
In `src/core_agents/critic.py`:
* **Zero Cost & Sub-Millisecond Speed**: Runs in `< 5ms` with **0 tokens consumed**.
* **4 Programmatic Gates**:
  1. *Gate 1 (AST Compilation)*: Validates clean syntax compilation.
  2. *Gate 2 (Negative Scan)*: Zero legacy tokens allowed (`@validator`, `@root_validator`, inner `class Config:`, `.dict()`, `.json()`, `from pydantic import BaseSettings`, `@app.on_event`).
  3. *Gate 3 (Positive Scan)*: Enforces required v2 constructs (`@field_validator` with `@classmethod`, `ConfigDict`, `SettingsConfigDict`, `model_dump()`).
  4. *Gate 4 (Runtime Sandbox)*: Instantiates models in an isolated sandbox with dynamic module mocking (`_MockModuleFinder`) to catch execution errors without failing on missing 3rd-party dependencies.

---

## ADR-04: Qdrant Dual-Mode vs. ChromaDB (The RocksDB Multi-Process Lock)

### Context & Failure Mode
Local AI development requires an embedded vector database that runs without Docker. ChromaDB uses an embedded RocksDB/SQLite storage engine. If a FastAPI server runs in Terminal A and holds the database directory open, running a test suite or ingestion script in Terminal B instantly crashes:
`RuntimeError: Storage folder is already accessed by another instance.`

### Options Considered
* **Option A (ChromaDB)**: Synchronous SQLite engine stalls FastAPI's event loop; crashes under concurrent test/server processes. **Rejected**.
* **Option B (Qdrant Dual-Mode - CHOSEN)**: Native Rust-backed vector store with non-blocking `AsyncQdrantClient`.

### Implementation
In `src/storage/qdrant_manager.py` and `copilot_api/main.py`:
* **Development/Production**: Connects to `./.qdrant_storage` with single-process ownership managed by FastAPI's `@asynccontextmanager` lifespan.
* **Hermetic Testing**: Automated tests initialize Qdrant in ephemeral memory (`:memory:`), guaranteeing zero lock collisions across parallel test workers.
* **Automatic Fallback Guardrail**: If a scoped topic search yields a top similarity score `< 0.55`, the manager automatically drops the filter and searches globally, preventing false-negative silence.

---

## ADR-05: The 6-Question Minimal Spanning Basis

### Context & Strategy
Evaluating a code migration copilot by running 500 tests that all test `@validator` on different variable names burns 100x the tokens without providing any new architectural information.

### The Decision
In `docs/eval_decision.md` and `tests/evals/golden_fixtures.py`, we identified the **6 fundamental, mutually exclusive breaking changes** that span the entire migration space:
1. `eval_01_config_dict`: Inner `class Config:` ➔ `model_config = ConfigDict(...)`
2. `eval_02_field_validator`: `@validator` ➔ `@field_validator` with `@classmethod` and `mode='before'`
3. `eval_03_model_validator`: Whole-model `@root_validator` ➔ `@model_validator`
4. `eval_04_serialization`: `.dict()` / `.json()` ➔ `.model_dump()` / `.model_dump_json()`
5. `eval_05_settings`: Package extraction ➔ `from pydantic_settings import BaseSettings, SettingsConfigDict`
6. `eval_06_fastapi_lifespan`: Deprecated `@app.on_event` ➔ Modern `@asynccontextmanager` Lifespan

If the Multi-Agent Supervisor-Critic passes these 6 orthogonal basis vectors, the system is fundamentally sound across all breaking interfaces.

---

## ADR-06: Dogfooding Pydantic v2 vs. Python `@dataclass`

### Context & The Paradox
Our copilot modernizes Pydantic code, but what should we use for our own internal agent data contracts (`DecomposedQuery`, `DocChunk`, `DraftResponse`)?
* **The "Knife Sharpening a Knife" Paradox**: If our own code uses Pydantic, what if our agent accidentally generates v1 syntax in its own codebase?

### Options Considered
* **Option A (Built-in `@dataclass`)**: Immune to Pydantic breaking changes, but requires 100+ lines of manual serialization boilerplate and custom JSON schemas for LLM structured outputs. **Rejected**.
* **Option B (Dogfooding Pydantic v2 - CHOSEN)**: Zero glue code. The Google GenAI SDK and Groq client natively accept Pydantic models for structured output (`response_schema`), and FastAPI routes validate payloads out-of-the-box.

### Mitigation
We permanently codified strict Pydantic v2 rules into `GEMINI.md` and project invariants, eliminating internal version drift.

---

## ADR-07: Dual Embedder Protocol & Real Offline Cached Embeddings

### Context & Sandbox Isolation
We need 768-dimensional dense vector embeddings for Qdrant. Calling external APIs (Google GenAI) during automated tests introduces network latency, rate limits (`HTTP 429`), and fails in offline CI sandboxes. Downloading local HuggingFace models (~400MB) bloats dependencies.

### Implementation
In `src/storage/embedder.py`:
* **`EmbedderProtocol`**: Defines a clean, uniform async interface.
* **`GeminiEmbedder`**: Calls Google GenAI `text-embedding-004` (768-dim) in live environments.
* **`CachedEmbedder`**: Operates against `src/storage/data/gemini_embeddings_cache.json`. Automated test suites execute against **100% real Google Gemini embeddings** completely offline in 0.005 seconds with zero network dependency.
* **`DeterministicMockEmbedder`**: Generates SHA256 unit-normalized vectors for hermetic unit testing.

---

## ADR-08: The "Ghost Ship" Disconnect Guardrail

### Context & Token Waste
In multi-agent systems, generating code across multiple iterations (`Decomposer ➔ Retriever ➔ Synthesizer ⟷ Critic`) takes compute time. If a user closes their browser tab or cancels a `curl` streaming request mid-flight, standard Python backends continue running the generator in the background, burning expensive cloud tokens for a client that is no longer listening.

### Implementation
In `src/copilot_api/routes.py`:
* Before every compute-heavy stage, the Server-Sent Events (SSE) generator checks:
  ```python
  if await request.is_disconnected():
      logger.info("Client disconnected. Aborting agent reflection loop to save tokens.")
      return
  ```
* Wrapped within Python 3.12 `asyncio.TaskGroup`. When the client disconnects, child agent tasks are cleanly cancelled immediately.
