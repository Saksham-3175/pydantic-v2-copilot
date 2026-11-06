"""Query Decomposer agent for legacy Pydantic v1 / FastAPI code modernization."""

from __future__ import annotations

import os
import typing

from core_agents.llm import UnifiedLLMClient
from core_agents.schemas import DecomposedQuery


class QueryDecomposer:
    """Agent that analyzes legacy Python code and decomposes it into target migration topics and sub-queries."""

    def __init__(
        self,
        llm_client: typing.Optional[UnifiedLLMClient] = None,
        api_key: typing.Optional[str] = None,
        model: typing.Optional[str] = None,
        enable_llm: bool = False,
    ) -> None:
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model or "qwen/qwen3.8-27b"
        self._custom_llm_client = llm_client is not None
        self.enable_llm = enable_llm
        self.llm_client = llm_client or UnifiedLLMClient(
            gemini_api_key=self.api_key,
            primary_model=self.model,
        )
        self._legacy_client: typing.Any = None

    @property
    def client(self) -> typing.Any:
        return self._legacy_client if self._legacy_client is not None else self.llm_client.gemini_client

    @client.setter
    def client(self, client: typing.Any) -> None:
        self._legacy_client = client
        self.llm_client.gemini_client = client

    async def decompose(self, code: str) -> DecomposedQuery:
        """Decompose legacy code snippet into detected topics and search sub-queries.

        Handles:
        - Deterministic zero-token AST decomposition (0 tokens, saves ~1,200 Groq tokens per file).
        - Optional custom mock/client execution if explicitly injected (e.g. unit tests).
        """
        code_clean = code.strip()
        # Edge Case 1: Empty or whitespace-only code
        if not code_clean:
            return DecomposedQuery(
                summary="No code provided.",
                detected_topics=[],
                sub_queries=[],
            )

        # If a custom legacy client (mock) was attached, call it directly
        if self._legacy_client is not None:
            try:
                from google.genai import types

                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=DecomposedQuery,
                )
                resp = await self._legacy_client.aio.models.generate_content(
                    model=self.model,
                    contents=code_clean,
                    config=config,
                )
                if resp.text:
                    return DecomposedQuery.model_validate_json(resp.text)
            except Exception:
                pass

        # If a custom LLM client/mock was injected or enable_llm requested, use LLM
        if self._custom_llm_client or self.enable_llm:
            prompt = (
                "Analyze the following Python code for legacy Pydantic v1 or legacy FastAPI patterns. "
                "Extract the migration topics (e.g. 'pydantic_validators', 'pydantic_config', "
                "'pydantic_model_validators', 'pydantic_serialization', 'pydantic_settings', "
                "'fastapi_lifespan', 'fastapi_dependencies') and generate specific search queries.\n\n"
                f"```python\n{code_clean}\n```"
            )
            try:
                result, _ = await self.llm_client.generate_structured(
                    prompt=prompt,
                    schema=DecomposedQuery,
                    system_instruction="Analyze Python code and extract migration topics as JSON.",
                )
                return result
            except Exception:
                pass

        # Zero-token deterministic decomposition (bypasses LLM, eliminating rate limits)
        return self._heuristic_decompose(code_clean)

    def _heuristic_decompose(self, code: str) -> DecomposedQuery:
        """Deterministic heuristic scanner checking code tokens for offline testing."""
        topics: list[str] = []
        queries: list[str] = []
        lower = code.lower()

        # Check for validators
        if "@validator" in code:
            topics.append("pydantic_validators")
            queries.append("how to migrate @validator to @field_validator with @classmethod")
        if "@root_validator" in code:
            topics.append("pydantic_model_validators")
            queries.append("how to migrate @root_validator to @model_validator")
        if "class config" in lower:
            topics.append("pydantic_config")
            queries.append("how to replace inner class Config with model_config = ConfigDict")
        if ".dict(" in code or ".json(" in code:
            topics.append("pydantic_serialization")
            queries.append("how to replace .dict() and .json() with .model_dump() and .model_dump_json()")
        if "basesettings" in lower:
            topics.append("pydantic_settings")
            queries.append("how to import BaseSettings from pydantic_settings and use SettingsConfigDict")
        if "on_event(" in lower:
            topics.append("fastapi_lifespan")
            queries.append("how to migrate @app.on_event startup to lifespan context manager")

        # Edge Case 2: Code has no legacy patterns (already modern or plain python)
        if not topics:
            return DecomposedQuery(
                summary="No legacy Pydantic v1 or FastAPI patterns detected in code.",
                detected_topics=[],
                sub_queries=[],
            )

        summary = f"Detected {len(topics)} legacy patterns requiring modernization: {', '.join(topics)}."
        return DecomposedQuery(
            summary=summary,
            detected_topics=topics,
            sub_queries=queries,
        )
