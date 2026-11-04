"""Code Synthesizer agent for modernizing Pydantic v1 code to Pydantic v2."""

from __future__ import annotations

import json
import os
import re
import typing

from groq import AsyncGroq

from core_agents.schemas import DocChunkPayload, DraftResponse


class CodeSynthesizer:
    """Agent that synthesizes modern Pydantic v2 code using Groq qwen/qwen3.8-27b and retrieved documentation."""

    def __init__(
        self,
        api_key: typing.Optional[str] = None,
        model: str = "qwen/qwen3.8-27b",
    ):
        key = api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            try:
                from dotenv import find_dotenv, load_dotenv

                load_dotenv(find_dotenv(usecwd=True))
                key = os.environ.get("GROQ_API_KEY")
            except Exception:
                pass
        self.api_key = key
        self.model = model
        self.client = AsyncGroq(api_key=self.api_key) if self.api_key else None

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

        # Live Groq API execution if client is configured and API key exists
        if self.client and self.api_key:
            try:
                return await self._call_groq_api(code_clean, context_chunks, feedback, iteration)
            except Exception:
                # Fallback to deterministic offline synthesizer on error/offline
                pass

        # Deterministic offline synthesizer (for hermetic testing)
        return self._heuristic_synthesize(code_clean, context_chunks, feedback, iteration)

    async def _call_groq_api(
        self,
        code: str,
        chunks: list[DocChunkPayload],
        feedback: typing.Optional[str],
        iteration: int,
    ) -> DraftResponse:
        docs_text = "\n\n---\n\n".join([f"### {c.breadcrumb}\n{c.content}" for c in chunks])
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
            f"LEGACY CODE TO MODERNIZE:\n```python\n{code}\n```"
            f"{feedback_prompt}\n\n"
            "Produce the modern Pydantic v2 implementation as valid JSON."
        )

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = resp.choices[0].message.content or "{}"
        data = json.loads(content)
        return DraftResponse(
            modern_code=data.get("modern_code", code),
            explanation=data.get("explanation", "Modernized to Pydantic v2."),
            changes_made=data.get("changes_made", []),
            iteration=iteration,
        )

    def _heuristic_synthesize(
        self,
        code: str,
        chunks: list[DocChunkPayload],
        feedback: typing.Optional[str],
        iteration: int,
    ) -> DraftResponse:
        """Deterministic rule-based code modernizer for fast offline testing."""
        modern = code
        changes = []

        # If feedback mentions dict(), replace it
        if feedback and (".dict()" in feedback or "dict" in feedback.lower()):
            modern = re.sub(r"\.dict\(", ".model_dump(", modern)
            changes.append("Replaced .dict() with .model_dump() based on Critic feedback")

        # ConfigDict transformation
        if "class Config:" in modern:
            need_import = "ConfigDict" not in modern
            config_pattern = re.compile(
                r"([ \t]*)class Config:\n((?:\1[ \t]+[^\n]+\n*)+)"
            )
            match = config_pattern.search(modern)
            if match:
                indent = match.group(1)
                body = match.group(2)
                attrs = []
                for line in body.strip().splitlines():
                    line = line.strip()
                    if "=" in line and not line.startswith("#"):
                        k, v = line.split("=", 1)
                        k = k.strip()
                        if k == "orm_mode":
                            k = "from_attributes"
                        attrs.append(f"{k}={v.strip()}")
                args_str = ", ".join(attrs)
                modern = (
                    modern[: match.start()]
                    + f"{indent}model_config = ConfigDict({args_str})\n"
                    + modern[match.end() :]
                )
            else:
                modern = re.sub(
                    r"class Config:\s+([a-zA-Z_0-9]+)\s*=\s*([^\n]+)",
                    r"model_config = ConfigDict(\1=\2)",
                    modern,
                )
                modern = re.sub(r"class Config:.*", "model_config = ConfigDict()", modern)
            if need_import or "ConfigDict" not in modern:
                modern = "from pydantic import ConfigDict\n" + modern
            changes.append("Replaced inner class Config with model_config = ConfigDict(...)")

        # Validator transformation
        if "@validator(" in modern:
            need_import = "field_validator" not in modern
            # Replace @validator with @field_validator and add @classmethod
            modern = re.sub(
                r"@validator\((.*?)\)\s*\n\s*def\s+",
                r"@field_validator(\1, mode='after')\n    @classmethod\n    def ",
                modern,
            )
            if need_import or "field_validator" not in modern:
                modern = "from pydantic import field_validator\n" + modern
            changes.append("Replaced @validator with @field_validator and added @classmethod")

        # Serialization
        if ".dict(" in modern:
            modern = re.sub(r"\.dict\(", ".model_dump(", modern)
            changes.append("Replaced .dict() with .model_dump()")

        return DraftResponse(
            modern_code=modern,
            explanation="Modernized Pydantic v1 constructs to Pydantic v2.",
            changes_made=changes,
            iteration=iteration,
        )
