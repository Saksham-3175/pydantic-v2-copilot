# Pydantic v2 Migration Copilot

An autonomous neuro-symbolic CLI tool engineered to modernize legacy Python codebases (Pydantic v1.x and legacy FastAPI patterns) to strict Pydantic v2 standards.

Combines **Deterministic Python AST Screening**, **Scoped Vector Retrieval (Qdrant)**, and **Multi-Provider LLMs (Groq / Gemini)** with a **4-Gate Verification Guardrail** that guarantees generated code compiles and instantiates without syntax errors or deprecated APIs.

---

## 1. Quickstart

### Installation
Install globally into your system `$PATH` using Astral `uv`:

```bash
uv tool install git+https://github.com/Saksham-3175/pydantic-v2-copilot.git
```
*(Or install inside an active virtual environment via `pip install git+https://github.com/Saksham-3175/pydantic-v2-copilot.git`)*

### CLI Usage

```bash
# 1. Fast AST scan across repository (<0.002s per file, 0 tokens)
copilot scan ./app/

# 2. Preview syntax-highlighted diffs safely without modifying files
copilot migrate ./app/ --diff

# 3. Apply verified migrations directly in-place
copilot migrate ./app/ --write

# 4. Migrate and run target repository pytest suite automatically
copilot migrate ./app/ --write --run-tests
```

---

## 2. Core Superpowers

| Feature | How It Works | Developer Benefit |
| :--- | :--- | :--- |
| **2-Stage AST Screener** | Uses native `ast.parse` to detect legacy v1 tokens before touching any AI. | Skips already-clean files instantly. **Saves 100% of LLM tokens** on untouched files. |
| **SHA-256 Content Cache** | Content-addressed cache stored in `~/.cache/pydantic_copilot/`. | Running `--write` after `--diff` applies modern code in `<0.01s` with **zero duplicate LLM calls**. |
| **4-Gate Critic Guardrail** | Validates syntax, scans for 0 legacy tokens, checks modern imports, and executes sandbox model instantiation. | Eliminates hallucinations. Code never touches disk unless it passes all 4 compiler gates. |
| **Deterministic Heuristic Engine** | Built-in AST rule transformer for mechanical transformations. | Runs 100% offline without API keys or internet connection. |

---

## 3. Battle-Tested on Production

The screener, migration pipeline, and Critic guardrails are validated directly against the production **[FastAPI RealWorld Example App (Conduit)](https://github.com/nsidnev/fastapi-realworld-example-app)** (72 files, legacy Pydantic v1 `class Config`, `@validator`, and old event patterns):

```bash
# Clone the real-world legacy target
git clone https://github.com/nsidnev/fastapi-realworld-example-app.git

# Screen the models directory in milliseconds
copilot scan ./fastapi-realworld-example-app/app/models/

# Modernize in-place with 4-gate verification
copilot migrate ./fastapi-realworld-example-app/app/models/ --write
```

---

## 4. Supported Migrations (Minimal Spanning Basis)

The Copilot handles the 6 canonical breaking changes between Pydantic v1 and v2:

| # | Legacy Pydantic v1 Syntax | Modern Pydantic v2 Standard |
| :-: | :--- | :--- |
| **1** | `class Config: orm_mode = True` | `model_config = ConfigDict(from_attributes=True)` |
| **2** | `@validator("field", pre=True)` | `@field_validator("field", mode="before")` + `@classmethod` |
| **3** | `@root_validator` | `@model_validator(mode="before" \| "after")` + `@classmethod` |
| **4** | `.dict()` and `.json()` | `.model_dump()` and `.model_dump_json()` |
| **5** | `from pydantic import BaseSettings` | `from pydantic_settings import BaseSettings, SettingsConfigDict` |
| **6** | `@app.on_event("startup")` | Modern `@asynccontextmanager` Lifespan handlers & `Annotated` |

---

## 5. How It Works

```text
[Target Legacy File]
         │
         ▼
[1. AST Screener] ──▶ (No legacy tokens found) ──▶ Skip (0 tokens, <0.002s)
         │
         ▼ (Legacy tokens detected)
[2. SHA-256 Cache] ──▶ (Hit from prior --diff) ──▶ Instant Apply (<0.01s)
         │
         ▼ (Cache miss)
[3. Scoped RAG + Synthesizer] ──▶ Groq (qwen/qwen3.8-27b) / Gemini Flash
         │
         ▼
[4. 4-Gate AST Critic] ──▶ Syntax + Negative Scan + Positive Scan + Runtime Sandbox
         │
         ├──▶ [Pass] ──▶ Write to disk & update SHA-256 cache
         └──▶ [Fail] ──▶ Self-Correction Reflection Loop (max 2 iterations)
```

> For deep architectural rationale, state-machine chunking, vector negation trap solutions, and full ADR records, see **[DECISIONS.md](DECISIONS.md)**.

---

## 6. Development & Testing

### Run Automated Hermetic Test Suite
All 114 unit, integration, and evaluation benchmark tests execute 100% offline in ~17 seconds:

```bash
uv run pytest tests/
```

### Run Evaluation Golden Benchmark
```bash
uv run python tests/evals/run_e2e_benchmark.py --tag baseline
```

### Optional: FastAPI Streaming Gateway
The core migration engine is also available via an async FastAPI gateway:

```bash
# Start local API server on port 8000
uv run uvicorn copilot_api.main:app --reload

# Health check
curl http://localhost:8000/health

# Real-time SSE migration stream
curl -N -X POST http://localhost:8000/v1/migrate/stream \
  -H "Content-Type: application/json" \
  -d '{"code": "class User(BaseModel):\n    class Config:\n        orm_mode = True"}'
```
