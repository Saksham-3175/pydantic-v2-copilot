"""Unit tests for UnifiedLLMClient cross-provider failover (Groq primary, Gemini secondary)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock
import pytest
from pydantic import BaseModel

from core_agents.decomposer import QueryDecomposer
from core_agents.llm import UnifiedLLMClient
from core_agents.schemas import DecomposedQuery, DraftResponse
from core_agents.synthesizer import CodeSynthesizer


class SimpleSchema(BaseModel):
    message: str
    code: int


@pytest.mark.asyncio
async def test_unified_llm_groq_primary_success() -> None:
    """Verifies that primary provider (Groq) is used when available and successful."""
    client = UnifiedLLMClient(groq_api_key="mock-groq-key", gemini_api_key="mock-gemini-key")

    mock_groq = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(content=json.dumps({"message": "Hello from Groq", "code": 200}))
        )
    ]
    mock_groq.chat.completions.create = AsyncMock(return_value=mock_resp)
    client.groq_client = mock_groq

    mock_gemini = MagicMock()
    mock_gemini.aio.models.generate_content = AsyncMock()
    client.gemini_client = mock_gemini

    result, provider = await client.generate_structured(
        prompt="Test prompt",
        schema=SimpleSchema,
    )

    assert provider == "groq"
    assert result.message == "Hello from Groq"
    assert result.code == 200
    mock_groq.chat.completions.create.assert_awaited_once()
    mock_gemini.aio.models.generate_content.assert_not_awaited()


@pytest.mark.asyncio
async def test_unified_llm_failover_to_gemini_on_groq_error() -> None:
    """Verifies that client automatically fails over to Gemini when Groq raises an exception."""
    client = UnifiedLLMClient(groq_api_key="mock-groq-key", gemini_api_key="mock-gemini-key")

    # Mock Groq failure (e.g. RateLimitError or NetworkError)
    mock_groq = MagicMock()
    mock_groq.chat.completions.create = AsyncMock(side_effect=RuntimeError("Groq 429 RateLimitError"))
    client.groq_client = mock_groq

    # Mock Gemini success
    mock_gemini = MagicMock()
    mock_gemini_resp = MagicMock()
    mock_gemini_resp.text = json.dumps({"message": "Hello from Gemini Fallback", "code": 200})
    mock_gemini.aio.models.generate_content = AsyncMock(return_value=mock_gemini_resp)
    client.gemini_client = mock_gemini

    result, provider = await client.generate_structured(
        prompt="Test prompt",
        schema=SimpleSchema,
    )

    assert provider == "gemini"
    assert result.message == "Hello from Gemini Fallback"
    mock_groq.chat.completions.create.assert_awaited_once()
    mock_gemini.aio.models.generate_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_unified_llm_both_fail_raises_runtime_error() -> None:
    """Verifies that RuntimeError is raised when all cloud providers fail."""
    client = UnifiedLLMClient(groq_api_key="mock-groq-key", gemini_api_key="mock-gemini-key")

    mock_groq = MagicMock()
    mock_groq.chat.completions.create = AsyncMock(side_effect=RuntimeError("Groq Down"))
    client.groq_client = mock_groq

    mock_gemini = MagicMock()
    mock_gemini.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("Gemini Down"))
    client.gemini_client = mock_gemini

    with pytest.raises(RuntimeError) as exc_info:
        await client.generate_structured(prompt="Test", schema=SimpleSchema)

    assert "All LLM providers failed" in str(exc_info.value)


@pytest.mark.asyncio
async def test_unified_llm_decomposer_integration() -> None:
    """Verifies QueryDecomposer uses UnifiedLLMClient with Groq primary and Gemini fallback."""
    mock_llm = MagicMock()
    mock_llm.generate_structured = AsyncMock(
        return_value=(
            DecomposedQuery(
                summary="Decomposed via Unified Client",
                detected_topics=["pydantic_config"],
                sub_queries=["migrate class Config to ConfigDict"],
            ),
            "groq",
        )
    )

    decomposer = QueryDecomposer(llm_client=mock_llm)
    result = await decomposer.decompose("class Config:\n    orm_mode = True")

    assert result.summary == "Decomposed via Unified Client"
    assert "pydantic_config" in result.detected_topics
    mock_llm.generate_structured.assert_awaited_once()


@pytest.mark.asyncio
async def test_unified_llm_synthesizer_integration() -> None:
    """Verifies CodeSynthesizer uses UnifiedLLMClient with Groq primary and Gemini fallback."""
    mock_llm = MagicMock()
    mock_llm.generate_structured = AsyncMock(
        return_value=(
            DraftResponse(
                modern_code="class ModernUser(BaseModel):\n    model_config = ConfigDict()",
                explanation="Modernized via Unified Client",
                changes_made=["Replaced Config with ConfigDict"],
                iteration=1,
            ),
            "groq",
        )
    )

    synthesizer = CodeSynthesizer(llm_client=mock_llm)
    result = await synthesizer.synthesize("class User(BaseModel): pass", context_chunks=[])

    assert "model_config = ConfigDict" in result.modern_code
    assert result.explanation == "Modernized via Unified Client"
    mock_llm.generate_structured.assert_awaited_once()
