"""Code Synthesizer agent for modernizing Pydantic v1 code to Pydantic v2."""

from __future__ import annotations

import json
import os
import re
import typing

from core_agents.llm import UnifiedLLMClient
from core_agents.schemas import DocChunkPayload, DraftResponse


def _prune_chunk_content(content: str, max_chars: int = 1200) -> str:
    """Trim verbose documentation content to stay well under LLM token limits while preserving code blocks."""
    if len(content) <= max_chars:
        return content
    trimmed = content[:max_chars]
    last_newline = trimmed.rfind("\n")
    if last_newline > max_chars // 2:
        trimmed = trimmed[:last_newline]
    return trimmed + "\n... [truncated for token efficiency]"


class CodeSynthesizer:
    """Agent that synthesizes modern Pydantic v2 code using unified LLM client (Groq primary, Gemini fallback)."""

    def __init__(
        self,
        llm_client: typing.Optional[UnifiedLLMClient] = None,
        api_key: typing.Optional[str] = None,
        model: typing.Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("GROQ_API_KEY")
        self.model = model or "qwen/qwen3.8-27b"
        self.llm_client = llm_client or UnifiedLLMClient(
            groq_api_key=self.api_key,
            primary_model=self.model,
        )
        self._legacy_client: typing.Any = None

    @property
    def client(self) -> typing.Any:
        return self._legacy_client if self._legacy_client is not None else self.llm_client.groq_client

    @client.setter
    def client(self, client: typing.Any) -> None:
        self._legacy_client = client
        self.llm_client.groq_client = client

    async def synthesize(
        self,
        legacy_code: str,
        context_chunks: list[DocChunkPayload],
        feedback: typing.Optional[str] = None,
        iteration: int = 1,
    ) -> DraftResponse:
        """Synthesize modernized code from legacy snippet and documentation context."""
        code_clean = legacy_code.strip()
        if not code_clean:
            return DraftResponse(
                modern_code="",
                explanation="No legacy code provided to modernize.",
                changes_made=[],
                iteration=iteration,
            )

        # 1. Try unified LLM client if API key configured or mock client attached
        if self.api_key or self._legacy_client is not None:
            pruned_chunks = context_chunks[:2]
            docs_text = "\n\n---\n\n".join(
                [f"### {c.breadcrumb}\n{_prune_chunk_content(c.content)}" for c in pruned_chunks]
            )
            feedback_prompt = (
                f"\n\nCRITICAL FIXES REQUIRED FROM PREVIOUS ATTEMPT:\n{feedback}\nYou MUST resolve every failure noted above."
                if feedback
                else ""
            )

            system_msg = (
                "You are an expert Python engineer modernizing legacy Pydantic v1 code to strict Pydantic v2 standards. "
                "Rules to strictly enforce:\n"
                "1. Replace @validator with @field_validator and ALWAYS include @classmethod decorator.\n"
                "2. Replace inner class Config with model_config = ConfigDict(...).\n"
                "3. Replace @root_validator with @model_validator(mode='before' or 'after').\n"
                "4. Replace .dict() with .model_dump() and .json() with .model_dump_json().\n"
                "5. Replace BaseSettings from pydantic with from pydantic_settings import BaseSettings, SettingsConfigDict.\n"
                "6. Replace @app.on_event with @asynccontextmanager lifespan.\n"
                'Return JSON matching: {"modern_code": "...", "explanation": "...", "changes_made": ["..."]}'
            )

            user_msg = (
                f"REFERENCE DOCUMENTATION:\n{docs_text}\n\n"
                f"LEGACY CODE TO MODERNIZE:\n```python\n{code_clean}\n```"
                f"{feedback_prompt}\n\n"
                "Produce the modern Pydantic v2 implementation as valid JSON."
            )

            try:
                draft, _ = await self.llm_client.generate_structured(
                    prompt=user_msg,
                    schema=DraftResponse,
                    system_instruction=system_msg,
                )
                draft.iteration = iteration
                return draft
            except Exception:
                # Fallback to deterministic offline synthesizer on error/offline
                pass

        # 2. Deterministic offline synthesizer (for hermetic testing)
        return self._heuristic_synthesize(code_clean, context_chunks, feedback, iteration)

    def _heuristic_synthesize(
        self,
        code: str,
        chunks: list[DocChunkPayload],
        feedback: typing.Optional[str],
        iteration: int,
    ) -> DraftResponse:
        """Deterministic rule-based code modernizer for fast offline testing."""
        modern = code
        changes: list[str] = []

        # If feedback mentions dict(), replace it based on Critic feedback
        if feedback and (".dict()" in feedback or "dict" in feedback.lower()):
            modern = re.sub(r"\.dict\(", ".model_dump(", modern)
            changes.append("Replaced .dict() with .model_dump() based on Critic feedback")

        # ---------------------------------------------------------------------
        # 1. FastAPI Lifespan (@app.on_event -> lifespan contextmanager)
        # ---------------------------------------------------------------------
        if "@app.on_event" in modern:
            startup_body = '    print("Application starting up")'
            shutdown_body = '    print("Application shutting down")'

            # Extract body if possible
            startup_match = re.search(r'@app\.on_event\("startup"\)\s*\nasync def \w+\(\):\n((?:[ \t]+[^\n]+\n*)+)', modern)
            if startup_match:
                startup_body = startup_match.group(1).rstrip()
            shutdown_match = re.search(r'@app\.on_event\("shutdown"\)\s*\nasync def \w+\(\):\n((?:[ \t]+[^\n]+\n*)+)', modern)
            if shutdown_match:
                shutdown_body = shutdown_match.group(1).rstrip()

            lifespan_code = (
                "from contextlib import asynccontextmanager\n"
                "from fastapi import FastAPI\n\n"
                "@asynccontextmanager\n"
                "async def lifespan(app: FastAPI):\n"
                f"{startup_body}\n"
                "    yield\n"
                f"{shutdown_body}\n\n"
                "app = FastAPI(lifespan=lifespan)\n"
            )
            modern = lifespan_code
            changes.append("Replaced deprecated @app.on_event startup/shutdown with @asynccontextmanager lifespan")
            return DraftResponse(
                modern_code=modern,
                explanation="Modernized FastAPI startup and shutdown events to modern asynccontextmanager lifespan.",
                changes_made=changes,
                iteration=iteration,
            )

        # ---------------------------------------------------------------------
        # 2. BaseSettings & SettingsConfigDict (pydantic -> pydantic_settings)
        # ---------------------------------------------------------------------
        is_settings = "BaseSettings" in modern or "env_prefix" in modern
        if "from pydantic import BaseSettings" in modern:
            modern = modern.replace(
                "from pydantic import BaseSettings",
                "from pydantic_settings import BaseSettings, SettingsConfigDict",
            )
            changes.append("Migrated BaseSettings import from pydantic to pydantic_settings")

        # ---------------------------------------------------------------------
        # 3. ConfigDict / SettingsConfigDict (inner class Config: removal)
        # ---------------------------------------------------------------------
        if re.search(r"class Config(?:\([^)]*\))?:", modern):
            need_config_dict_import = (not is_settings) and ("ConfigDict" not in modern)
            config_cls = "SettingsConfigDict" if is_settings else "ConfigDict"
            config_pattern = re.compile(r"([ \t]*)class Config(?:\([^)]*\))?:\n((?:\1[ \t]+[^\n]+\n*)+)")
            match = config_pattern.search(modern)
            if match:
                indent = match.group(1)
                body = match.group(2)
                attrs: list[str] = []
                for line in body.strip().splitlines():
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip()
                        if k == "orm_mode":
                            k = "from_attributes"
                        elif k == "allow_mutation":
                            k = "frozen"
                            v = "True" if v.lower() == "false" else "False"
                        elif k == "allow_population_by_field_name":
                            k = "populate_by_name"
                        attrs.append(f"{k}={v}")
                if config_cls == "SettingsConfigDict" and not any(a.startswith("extra=") for a in attrs):
                    attrs.append('extra="ignore"')
                args_str = ", ".join(attrs)
                modern = (
                    modern[: match.start()]
                    + f"{indent}model_config = {config_cls}({args_str})\n"
                    + modern[match.end() :]
                )
            else:
                modern = re.sub(
                    r"class Config(?:\([^)]*\))?:\s+([a-zA-Z_0-9]+)\s*=\s*([^\n]+)",
                    rf"model_config = {config_cls}(\1=\2)",
                    modern,
                )
                modern = re.sub(r"class Config(?:\([^)]*\))?:.*", rf"model_config = {config_cls}()", modern)

            if need_config_dict_import:
                if "from pydantic import " in modern:
                    modern = re.sub(r"from pydantic import ([^\n]+)", r"from pydantic import \1, ConfigDict", modern, count=1)
                else:
                    modern = "from pydantic import ConfigDict\n" + modern
            changes.append(f"Replaced inner class Config with model_config = {config_cls}(...)")

        # ---------------------------------------------------------------------
        # 4. Field Validator (@validator -> @field_validator with @classmethod)
        # ---------------------------------------------------------------------
        if "@validator(" in modern:
            def _replace_validator(m: re.Match[str]) -> str:
                args = m.group(1).strip()
                if "pre=True" in args:
                    args = re.sub(r"\bpre=True\b", "mode='before'", args)
                elif "mode=" not in args:
                    args = f"{args}, mode='after'" if args else "mode='after'"
                # Clean up duplicate commas or whitespace
                args = re.sub(r",\s*,", ",", args).strip(" ,")
                return f"@field_validator({args})\n    @classmethod\n    def "

            modern = re.sub(
                r"@validator\((.*?)\)\s*\n\s*def\s+",
                _replace_validator,
                modern,
            )
            # Fix imports
            modern = re.sub(r"\bfrom pydantic import ([^\n]*)\bvalidator\b([^\n]*)", r"from pydantic import \1field_validator\2", modern)
            if "field_validator" not in modern:
                modern = "from pydantic import field_validator\n" + modern
            # Clean up dangling commas in imports e.g. "import BaseModel, "
            modern = re.sub(r",\s*,", ", ", modern)
            modern = re.sub(r",\s*\n", "\n", modern)
            changes.append("Replaced @validator with @field_validator and added @classmethod")

        # ---------------------------------------------------------------------
        # 5. Model Validator (@root_validator -> @model_validator)
        # ---------------------------------------------------------------------
        if "@root_validator" in modern:
            modern = re.sub(
                r"@root_validator(?:\(.*?\))?\s*\n\s*def\s+",
                r"@model_validator(mode='before')\n    @classmethod\n    def ",
                modern,
            )
            # Fix imports
            modern = re.sub(r"\bfrom pydantic import ([^\n]*)\broot_validator\b([^\n]*)", r"from pydantic import \1model_validator\2", modern)
            if "model_validator" not in modern:
                modern = "from pydantic import model_validator\n" + modern
            modern = re.sub(r",\s*,", ", ", modern)
            modern = re.sub(r",\s*\n", "\n", modern)
            changes.append("Replaced @root_validator with @model_validator(mode='before') and @classmethod")

        # ---------------------------------------------------------------------
        # 6. Serialization (.dict() -> .model_dump() and .json() -> .model_dump_json())
        # ---------------------------------------------------------------------
        if ".dict(" in modern:
            modern = re.sub(r"\.dict\(", ".model_dump(", modern)
            changes.append("Replaced .dict() with .model_dump()")
        if ".json(" in modern:
            modern = re.sub(r"\.json\(", ".model_dump_json(", modern)
            changes.append("Replaced .json() with .model_dump_json()")

        # Clean duplicate import lines if any
        lines = modern.splitlines()
        seen_imports: set[str] = set()
        cleaned_lines: list[str] = []
        for line in lines:
            line_s = line.strip()
            if line_s.startswith("from pydantic") or line_s.startswith("import pydantic"):
                if line_s in seen_imports:
                    continue
                seen_imports.add(line_s)
            cleaned_lines.append(line)
        modern = "\n".join(cleaned_lines)

        return DraftResponse(
            modern_code=modern,
            explanation="Modernized Pydantic v1 constructs to Pydantic v2.",
            changes_made=changes,
            iteration=iteration,
        )
