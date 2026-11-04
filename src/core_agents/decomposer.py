"""Query Decomposer agent for legacy Pydantic v1 / FastAPI code modernization."""

from __future__ import annotations

import os
import typing
from google import genai
from google.genai import types

from core_agents.schemas import DecomposedQuery


class QueryDecomposer:
    """Agent that analyzes legacy Python code and decomposes it into target migration topics and sub-queries."""

    def __init__(self, api_key: typing.Optional[str] = None, model: str = "gemini-3.8-flash"):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.model = model
        self.client = genai.Client(api_key=self.api_key) if self.api_key else None

    async def decompose(self, code: str) -> DecomposedQuery:
        """Decompose legacy code snippet into detected topics and search sub-queries.

        Handles both:
        - Live Gemini API execution via response_schema=DecomposedQuery when client is present.
        - Deterministic offline heuristic fallback for tests/offline environments.
        """
        code_clean = code.strip()
        # Edge Case 1: Empty or whitespace-only code
        if not code_clean:
            return DecomposedQuery(
                summary="No code provided.",
                detected_topics=[],
                sub_queries=[],
            )

        # If live client available, use gemini-3.8-flash with structured output
        if self.client and (self.api_key or os.environ.get("GEMINI_API_KEY")):
            try:
                prompt = (
                    "Analyze the following Python code for legacy Pydantic v1 or legacy FastAPI patterns. "
                    "Extract the migration topics (e.g. 'pydantic_validators', 'pydantic_config', "
                    "'pydantic_model_validators', 'pydantic_serialization', 'pydantic_settings', "
                    "'fastapi_lifespan', 'fastapi_dependencies') and generate specific search queries.\n\n"
                    f"```python\n{code_clean}\n```"
                )
                response = await self.client.aio.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        response_schema=DecomposedQuery,
                    ),
                )
                if response.text:
                    return DecomposedQuery.model_validate_json(response.text)
            except Exception:
                # Fallback to heuristic parser on network/API failure
                pass

        # Heuristic offline decomposition (for fast hermetic testing & offline mode)
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
