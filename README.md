# Pydantic v2 Migration Copilot

A high-precision, multi-agent RAG system engineered to modernize legacy Python codebases (Pydantic v1.x and legacy FastAPI patterns) to strict Pydantic v2 standards. 

Built with **FastAPI**, **Qdrant Vector Database**, **Groq API (`qwen/qwen3.8-27b`)**, **Google GenAI Embeddings (`gemini-embedding-001`)**, and **Deterministic Python AST Verification**.

> For in-depth architectural trade-off records and design rationales, see [DECISIONS.md](DECISIONS.md).

---

## 1. System Architecture

The Copilot follows a hierarchical multi-agent supervisor-critic pattern backed by scoped vector retrieval and deterministic static code validation:

```text
[User Legacy Code Input]
           │
           ▼
[FastAPI Gateway] (Lifespan Context Manager & SSE Streaming)
   ├── POST /v1/migrate        (Synchronous Batch)
   └── POST /v1/migrate/stream (Real-Time Server-Sent Events)
           │
           ▼
[1. Query Decomposer Agent] ──▶ Extracts topic taxonomies & sub-queries (Gemini Flash / Heuristic)
           │
           ▼
[2. Scoped Retriever Agent] ──▶ Qdrant dense vector search with topic payload filtering
           │                    ┌──────────────────────────────────────────────┐
           │                    │ Automatic Fallback Rule:                     │
           │                    │ If top score < 0.55, drops filter & searches │
           │                    │ globally across full corpus.                 │
           │                    └──────────────────────────────────────────────┘
           ▼
[3. Code Synthesizer Agent] ──▶ Generates modern Pydantic v2 code (Groq qwen/qwen3.8-27b)
           │
           ▼
[4. AST & Rule Critic]      ──▶ 4-Gate Deterministic Verification (Zero LLM Judge)
           │                    ├── Gate 1: ast.parse() syntax validation
           │                    ├── Gate 2: Negative Invariant Scan (0 legacy v1 tokens)
           │                    ├── Gate 3: Positive Invariant Scan (ConfigDict, @classmethod)
           │                    └── Gate 4: Runtime Model Instantiation Sandbox
           │
           ├──▶ [Pass] ───────▶ Stream verified modernization result to client
           └──▶ [Fail] ───────▶ Self-Correction Reflection Loop back to [3. Synthesizer]
                                (carries exact gate failure lines & suggested fixes; max_iter=2)
```

---

## 2. The MVP Scope: Minimal Spanning Basis

Rather than attempting to superficially index all 50,000 words of Pydantic documentation, this MVP focuses with extreme precision on the **6 canonical breaking changes** (the minimal spanning basis) that account for the vast majority of real-world migration friction:

1. `class Config:` ➔ `model_config = ConfigDict(...)`
2. `@validator` (field) ➔ `@field_validator` with `@classmethod` and `mode='after'|'before'`
3. `@root_validator` (model) ➔ `@model_validator(mode="before"|"after")`
4. `.dict()` / `.json()` export ➔ `.model_dump()` / `.model_dump_json()`
5. `pydantic.BaseSettings` extraction ➔ `from pydantic_settings import BaseSettings, SettingsConfigDict`
6. FastAPI `@app.on_event("startup")` ➔ Modern `@asynccontextmanager` Lifespan handlers & `Annotated` dependencies

---

## 3. Developer CLI Quickstart (`copilot`)

The Copilot includes an in-process developer CLI powered by Typer and Rich:

```bash
# 1. Install CLI globally into your system $PATH:
uv tool install ./dist/pydantic_v2_copilot-0.1.0-py3-none-any.whl --force
# (or in any active virtual environment: pip install ./dist/pydantic_v2_copilot-0.1.0-py3-none-any.whl)

# 2. Pre-flight AST Scan (screens clean files in <0.002ms):
copilot scan ./app/models/

# 3. Safe Dry-Run (preview syntax-highlighted diffs without touching disk):
copilot migrate ./app/models/ --diff

# 4. In-Place Migration (applies verified Pydantic v2 code directly):
copilot migrate ./app/models/ --write

# 5. Full Autonomous Pipeline (migrate & run target pytest suite):
copilot migrate ./app/models/ --write --run-tests
```

### Key CLI Features
* **2-Stage AST Screener**: Scans repositories in milliseconds; skips files with 0 legacy tokens to save 100% of LLM calls.
* **Content-Addressed SHA-256 Cache**: Running `--write` after `--diff` applies the verified code in `<0.01s` with **zero duplicate LLM calls**.
* **4-Gate Critic Guardrail**: Every file is compiled and tested across 4 gates before writing to disk.

### Real-World Production Benchmark
All real-world CLI migrations, AST screening benchmarks, and multi-file verification tests were executed directly against the production [FastAPI RealWorld Example App (Conduit)](https://github.com/nsidnev/fastapi-realworld-example-app) repository (covering 72 Python files, legacy Pydantic v1 `class Config`, `@validator`, and legacy FastAPI event patterns):
```bash
# 1. Clone the real-world legacy target:
git clone https://github.com/nsidnev/fastapi-realworld-example-app.git

# 2. Fast scan across the repository:
copilot scan ./fastapi-realworld-example-app/app/models/

# 3. Modernize in-place with Critic verification:
copilot migrate ./fastapi-realworld-example-app/app/models/ --write
```

---

## 3. Storage Engineering & The RAG Pipeline

### 3.1 State-Machine Chunker (The Indentation Guard)
Standard RAG systems rely on sliding token windows, which indiscriminately slice code blocks in half, resulting in fatal `IndentationError` during retrieval. 

The `MarkdownHeaderChunker` (`src/storage/chunker.py`) uses a line-by-line state machine tracking code fence boundaries (`in_code_block: bool`). Code blocks (` ```python ... ``` `) are treated as indivisible atomic units and are guaranteed never to be split across chunk boundaries.

### 3.2 The Negation Trap Solution
In migration documentation, text often reads: *"Deprecated: Do NOT use `@validator`"*. A naive vector embedding will embed the legacy syntax, causing vector searches for legacy code to score deceptively high.

The system programmatically decouples:
* **`search_text`**: Stripped of legacy code snippets and comments, containing only modern v2 target signatures and breadcrumbs. Embeddings represent the *desired modern target*.
* **`payload_content`**: Retains the rich Before/After examples and full explanatory context for the Synthesizer agent and human reviewer.

### 3.3 Official Documentation Sources
The bundled corpus snapshots in `src/storage/data/` were sourced directly from the official upstream repositories:
* **Pydantic V2 Migration Guide**: [https://docs.pydantic.dev/latest/migration/](https://docs.pydantic.dev/latest/migration/)
* **Pydantic Settings Documentation**: [https://docs.pydantic.dev/latest/concepts/pydantic_settings/](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)
* **FastAPI Lifespan Events**: [https://fastapi.tiangolo.com/advanced/events/](https://fastapi.tiangolo.com/advanced/events/)
* **FastAPI Annotated Dependencies**: [https://fastapi.tiangolo.com/tutorial/dependencies/](https://fastapi.tiangolo.com/tutorial/dependencies/)

### 3.4 Dual-Layer Vector Persistence & Offline Real Cache
Embeddings are generated using Google GenAI `gemini-embedding-001` (768 dimensions, unit $L_2$ normalized):
1. **Live On-Disk Qdrant (`./.qdrant_storage`)**: Persists points and Rust-backed HNSW graphs for live runtime search.
2. **Offline Vector Snapshot (`src/storage/data/gemini_embeddings_cache.json`)**: Real 768-float Gemini vectors are saved into a lightweight JSON snapshot. This allows automated tests and CI/CD pipelines to run against **100% real Google Gemini embeddings** completely offline in 0.005 seconds with zero API costs.

---

## 4. Scoped Vector Retrieval & The Automatic Fallback Guardrail

Managed by `QdrantStorageManager` (`src/storage/qdrant_manager.py`), search queries leverage scoped payload filtering with an intelligent safety net:

```text
┌────────────────────────────────────────────────────────┐
│          THE AUTOMATIC FALLBACK GUARDRAIL              │
├────────────────────────────────────────────────────────┤
│ 1. Scoped Search:                                      │
│    When a query topic is identified (e.g.              │
│    "pydantic_validators"), Qdrant filters the graph    │
│    during traversal in Rust:                           │
│    models.Filter(topic == "pydantic_validators")       │
│                                                        │
│ 2. The False-Negative Hazard:                          │
│    If a query has vocabulary drift or spans multiple   │
│    topics, strict filtering can cause "blindness"      │
│    (returning empty results or irrelevant weak hits).  │
│                                                        │
│ 3. The Fallback Rule:                                  │
│    If the top retrieved score in the filtered aisle is │
│    < 0.55 (a doubtful match):                          │
│    ➔ The manager automatically DROPS the topic filter. │
│    ➔ It immediately re-executes a global search.       │
│    ➔ The user NEVER gets an empty false-negative.      │
└────────────────────────────────────────────────────────┘
```

---

## 5. The Multi-Agent Team (`src/core_agents/`)

Each agent in the core pipeline has high cohesion and loose coupling, communicating strictly via Pydantic v2 data contracts (`src/core_agents/schemas.py`):

| Agent | Module | Primary Engine | Role & Fallback |
| :--- | :--- | :--- | :--- |
| **`QueryDecomposer`** | `decomposer.py` | Gemini Flash (`gemini-3.8-flash`) | Analyzes legacy code to detect migration topics and generates targeted search sub-queries. Includes offline keyword heuristic fallback. |
| **`ScopedRetriever`** | `retriever.py` | Async Qdrant Client + Embedder | Performs scoped vector retrieval against topic partitions with automatic global fallback on low scores. |
| **`CodeSynthesizer`** | `synthesizer.py` | Groq API (`qwen/qwen3.8-27b`) | Modernizes code given retrieved context chunks. Implements dynamic reflection on AST Critic feedback traces. Includes offline rule-based fallback. |
| **`ASTCritic`** | `critic.py` | Native Python AST (`ast.parse`) | Deterministic static analyzer enforcing 4 verification gates (zero legacy v1 tokens, required v2 imports/decorators, runtime sandbox). **Zero LLM tokens, 100% offline.** |
| **`MigrationOrchestrator`** | `orchestrator.py` | Python Async Supervisor | Coordinates linear-to-cyclic execution (`Decomposer ➔ Retriever ➔ Synthesizer ⟷ Critic`). Manages the multi-turn self-correction reflection loop (`max_iterations = 2`). |

---

## 6. FastAPI Application & SSE Streaming Gateway (`src/copilot_api/`)

The application gateway is built on modern FastAPI and async design patterns:

### 6.1 Lifespan Management (`@asynccontextmanager`)
Replaces deprecated `@app.on_event("startup")` and `@app.on_event("shutdown")`. On startup, it preheats the local Qdrant collection, initializes the embedder cache, and binds the `MigrationOrchestrator` to `app.state`. On shutdown, database connections are safely terminated.

### 6.2 The "Ghost Ship" Disconnect Guardrail
Multi-agent LLM loops can burn expensive tokens if a user disconnects mid-flight. The Server-Sent Events (SSE) generator checks:
```python
if await request.is_disconnected():
    logger.info("Client disconnected. Aborting agent reflection loop to save tokens.")
    return
```
This check is evaluated before every heavy stage (decomposition, vector retrieval, Groq synthesis, and critic verification).

---

## 7. CLI Runbook & Testing Matrix

### 7.1 Running the Automated Test Suite
All automated unit, integration, and evaluation tests execute 100% hermetically offline in ~40 seconds using Astral `uv`:

```bash
# Run the complete test suite (92 tests: 91 passed, 1 skipped)
uv run pytest tests/ -v
```

### Complete Test Matrix:
* **`tests/evals/test_golden_evals.py` (6 tests)**: The 6-Question Golden Benchmark (Minimal Spanning Basis) verifying all 4 AST & runtime gates.
* **`tests/test_agent_schemas.py` (5 tests)**: Inter-agent Pydantic v2 data contracts.
* **`tests/test_critic.py` (13 tests)**: 4 deterministic AST verification gates and failure detection.
* **`tests/test_decomposer.py` (7 tests)**: Topic decomposition, search query formulation, empty string & comments edge cases.
* **`tests/test_retriever.py` (4 tests)**: Scoped topic retrieval, chunk deduplication, and automatic fallback.
* **`tests/test_synthesizer.py` (5 tests)**: Groq `qwen/qwen3.8-27b` integration, reflection on critic feedback, and offline heuristics.
* **`tests/test_orchestrator.py` (7 tests)**: End-to-end migration supervisor, reflection loop retry, iteration exhaustion, and edge cases.
* **`tests/test_api.py` (10 tests)**: FastAPI routes (`GET /health`, `POST /v1/migrate`, `POST /v1/migrate/stream`), SSE event lifecycle, and disconnect handling.
* **`tests/test_unified_llm.py` (5 tests)**: Multi-provider failover (Groq primary, Gemini secondary).
* **Storage Tests (`tests/test_chunker.py`, `test_embedder.py`, `test_ingest.py`, `test_qdrant_manager.py`, `test_real_embeddings.py`) (30 tests)**: State-machine chunking, vector normalization, batch ingestion, and Qdrant lifecycle.

---

### 7.2 Starting the FastAPI Server
```bash
# Start local API development server on port 8000
uv run uvicorn copilot_api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 7.3 Interacting with the API via `curl`

#### 1. Heartbeat Health Check
```bash
curl http://localhost:8000/health
```

#### 2. Synchronous Migration (`POST /v1/migrate`)
```bash
curl -X POST http://localhost:8000/v1/migrate \
  -H "Content-Type: application/json" \
  -d '{
    "code": "from pydantic import BaseModel, validator\n\nclass User(BaseModel):\n    name: str\n\n    class Config:\n        orm_mode = True\n\n    @validator(\"name\")\n    def check_name(cls, v):\n        return v.title()",
    "max_iterations": 2
  }'
```

#### 3. Real-Time Streaming Migration (`POST /v1/migrate/stream`)
```bash
curl -N -X POST http://localhost:8000/v1/migrate/stream \
  -H "Content-Type: application/json" \
  -d '{
    "code": "from pydantic import BaseModel, validator\n\nclass User(BaseModel):\n    name: str\n    @validator(\"name\")\n    def check_name(cls, v): return v",
    "max_iterations": 2
  }'
```

---

### 7.4 Running Ingestion Manually
```bash
# Ingest curated documentation into local disk Qdrant using Gemini API
uv run python -m storage.ingest --recreate

# Run dry-run without writing to disk
uv run python -m storage.ingest --dry-run
```

---

## 8. Evaluation-Driven Development & Golden Benchmark

The project incorporates an **Evaluation-Driven Development (EDD)** framework that continuously measures accuracy against the **6-Question Golden Benchmark (The Minimal Spanning Basis)**.

### 8.1 The 6 Orthogonal Basis Vectors
* **`eval_01_config_dict`**: Migrates inner `class Config:` (`orm_mode=True`, `allow_mutation=False`) ➔ `model_config = ConfigDict(from_attributes=True, frozen=True)`.
* **`eval_02_field_validator`**: Migrates `@validator('name', pre=True)` ➔ `@field_validator('name', mode='before')` with `@classmethod`.
* **`eval_03_model_validator`**: Migrates whole-model `@root_validator` ➔ `@model_validator(mode='before')` with `@classmethod`.
* **`eval_04_serialization`**: Migrates export methods `.dict()` and `.json()` ➔ `.model_dump()` and `.model_dump_json()`.
* **`eval_05_settings`**: Migrates `BaseSettings` extraction ➔ `from pydantic_settings import BaseSettings, SettingsConfigDict`.
* **`eval_06_fastapi_lifespan`**: Migrates deprecated `@app.on_event` ➔ `@asynccontextmanager` FastAPI lifespan handlers.

### 8.2 The 4 Deterministic Verification Gates
1. **Gate 1 (AST Syntax)**: `ast.parse()` verifies valid compilation with zero syntax errors.
2. **Gate 2 (Negative Invariant Scan)**: 0 occurrences of deprecated v1 tokens (`@validator`, `@root_validator`, inner `class Config:`, `.dict()`, `.json()`, `from pydantic import BaseSettings`, `@app.on_event`).
3. **Gate 3 (Positive Invariant Scan)**: Required modern v2 tokens present (`@field_validator` with `@classmethod`, `ConfigDict`, `SettingsConfigDict`, `model_dump()`, `lifespan`).
4. **Gate 4 (Runtime Sandbox)**: `exec()` dynamically instantiates models in an isolated sandbox with dynamic module mocking (`_MockModuleFinder`).

### 8.3 Running the Evaluation Benchmark
```bash
# Run the 4-gate automated evaluation test suite
uv run pytest tests/evals/test_golden_evals.py -v

# Run the standalone E2E benchmark CLI and update markdown report
uv run python tests/evals/run_e2e_benchmark.py --tag baseline
```

### 8.4 Persistent Run Telemetry (`tests/evals/results/`)
Benchmark executions write structured JSON telemetry to disk:
* **Run Folder**: `tests/evals/results/run_<timestamp>_<tag>/`
* **Individual Eval Records**: `eval_01_config_dict.json` through `eval_06_fastapi_lifespan.json` containing exact input/output diffs, latency, and gate verification statuses.
* **Run Summary**: `summary.json` with aggregate pass rates and latency statistics.
* **Historical Ledger**: `tests/evals/results/history_manifest.json` tracking pass rate trends over time.
