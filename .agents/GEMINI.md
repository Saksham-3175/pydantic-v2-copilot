# Antigravity Rules & Guidelines

## Agent Autonomy & Execution Boundary
- **Explicit Permission Required**: Never create new code files, edit existing source code, or execute mutating/daemon shell commands without explaining the proposed action and receiving the user's explicit green light first. Always pause and ask for confirmation before executing.

## Visuals, Flowcharts & Diagrams
- **No Complex Mermaid in Terminal Chat**: Never output complex Mermaid diagrams (such as `sequenceDiagram` or wide horizontal charts) directly in terminal chat. The CLI renders them as unreadable ASCII art where text wraps unpredictably and arrowheads slice through words.
- **Preferred Visual Formatting**:
  - Use clean, vertical Unicode box pipelines (e.g., `[Step 1] ──▶ [Step 2] ──▶ [Step 3]`).
  - Use formatted Markdown tables for comparisons and matrices.
  - Use structured, nested bullet trees and numbered pipelines.
  - If Mermaid diagrams are necessary, only embed them in persistent artifact documents (`.md` in the brain/artifact folder) viewed in GUI renderers, never in raw terminal output.

## Architecture & System Reference
- **System Truth**: Consult `pydantic-v2-copilot/PROJECT.md` when designing, refactoring, or querying storage schemas, RAG invariants, and agent contracts.

## Python Toolchain & Package Management
- **Python Version**: Strictly Python `>=3.12`.
- **Tooling Engine**: Always use Astral `uv` (`uv add`, `uv run`, `uv sync`). Never execute bare `pip install`.
- **Workspace Architecture**: Maintain isolated packages under `apps/` and `libs/` using `[tool.uv.workspace]`.

## Pydantic v2 & FastAPI Standards (Dogfooding)
- **Model Configuration**: Use `model_config = ConfigDict(...)`. Never use legacy inner `class Config:`.
- **Field Validation**: Use `@field_validator` with `@classmethod` and `ValidationInfo`. Never use deprecated `@validator`.
- **Model Serialization**: Use `.model_dump()` and `.model_dump_json()`. Never use `.dict()` or `.json()`.
- **Settings Extraction**: Import `BaseSettings` and `SettingsConfigDict` from `pydantic_settings`. Never import `BaseSettings` from `pydantic`.
- **Dependency Injection**: Use `typing.Annotated[DepType, Depends(get_dep)]`. Never use default parameter assignment (`dep: DepType = Depends(...)`).
- **Async & Concurrency**: Async-first for all FastAPI route handlers and Qdrant database clients.
