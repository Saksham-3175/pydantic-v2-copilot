"""Unit tests for CodeSynthesizer agent covering offline heuristics, reflection, and Groq API calls."""

import json
import os
from unittest.mock import AsyncMock, MagicMock
import pytest

from core_agents.schemas import DocChunkPayload, DraftResponse
from core_agents.synthesizer import CodeSynthesizer


@pytest.mark.asyncio
async def test_synthesizer_happy_path_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests heuristic transformation of legacy code with @validator and class Config:."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    synthesizer = CodeSynthesizer(api_key=None)

    legacy_code = """
from pydantic import BaseModel, validator

class UserModel(BaseModel):
    name: str

    class Config:
        orm_mode = True

    @validator("name")
    def name_must_be_capitalized(cls, v: str) -> str:
        return v.title()
"""
    chunks = [
        DocChunkPayload(
            chunk_id="chunk-1",
            topic="pydantic_validators",
            breadcrumb="Validators > Field Validator",
            content="Use @field_validator with @classmethod",
        ),
        DocChunkPayload(
            chunk_id="chunk-2",
            topic="pydantic_config",
            breadcrumb="Configuration > ConfigDict",
            content="Use model_config = ConfigDict(from_attributes=True)",
        ),
    ]

    result = await synthesizer.synthesize(legacy_code, chunks)
    assert isinstance(result, DraftResponse)
    assert "@field_validator" in result.modern_code
    assert "@classmethod" in result.modern_code
    assert "model_config = ConfigDict" in result.modern_code
    assert "@validator" not in result.modern_code
    assert "class Config:" not in result.modern_code
    assert len(result.changes_made) >= 2
    assert any("Config" in c for c in result.changes_made)
    assert any("validator" in c for c in result.changes_made)
    assert result.iteration == 1


@pytest.mark.asyncio
async def test_synthesizer_reflection_on_critic_feedback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Feeds feedback complaining about .dict(), asserts synthesized code replaces .dict() with .model_dump()."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    synthesizer = CodeSynthesizer(api_key=None)

    legacy_code = """
class DataHolder:
    def dump(self):
        return self.dict()
"""
    feedback = "Gate 2 Failure: Legacy method .dict() is deprecated in Pydantic v2. Use .model_dump()."
    result = await synthesizer.synthesize(
        legacy_code=legacy_code,
        context_chunks=[],
        feedback=feedback,
        iteration=2,
    )
    assert isinstance(result, DraftResponse)
    assert ".model_dump(" in result.modern_code
    assert ".dict(" not in result.modern_code
    assert "Replaced .dict() with .model_dump() based on Critic feedback" in result.changes_made
    assert result.iteration == 2


@pytest.mark.asyncio
async def test_synthesizer_empty_code_edge_case(monkeypatch: pytest.MonkeyPatch) -> None:
    """Feeds empty string and whitespace-only string, returns clean empty DraftResponse."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    synthesizer = CodeSynthesizer(api_key=None)

    empty_res = await synthesizer.synthesize("", [])
    assert empty_res.modern_code == ""
    assert empty_res.explanation == "No legacy code provided to modernize."
    assert empty_res.changes_made == []
    assert empty_res.iteration == 1

    ws_res = await synthesizer.synthesize("   \n\t  \n  ", [], iteration=3)
    assert ws_res.modern_code == ""
    assert ws_res.explanation == "No legacy code provided to modernize."
    assert ws_res.changes_made == []
    assert ws_res.iteration == 3


@pytest.mark.asyncio
async def test_synthesizer_groq_api_call_or_mock() -> None:
    """If GROQ_API_KEY is present, tests live completion or offline fallback with qwen/qwen3.8-27b and asserts modern_code is non-empty."""
    legacy_code = """
class LegacyUser(BaseModel):
    name: str

    class Config:
        orm_mode = True
"""
    chunks = [
        DocChunkPayload(
            chunk_id="config-chunk",
            topic="pydantic_config",
            breadcrumb="Config",
            content="Use model_config = ConfigDict(from_attributes=True)",
        )
    ]

    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        synthesizer = CodeSynthesizer(api_key=api_key, model="qwen/qwen3.8-27b")
        res = await synthesizer.synthesize(legacy_code, chunks)
        assert isinstance(res, DraftResponse)
        assert res.modern_code != ""
        assert len(res.modern_code) > 0
    else:
        synthesizer = CodeSynthesizer(api_key="mock-groq-key")
        mock_response = MagicMock()
        mock_response.choices = [
            MagicMock(
                message=MagicMock(
                    content='{"modern_code": "model_config = ConfigDict(from_attributes=True)", "explanation": "Migrated Config", "changes_made": ["ConfigDict"]}'
                )
            )
        ]
        mock_client = MagicMock()
        mock_client.chat.completions.create = AsyncMock(return_value=mock_response)
        synthesizer.client = mock_client

        res = await synthesizer.synthesize(legacy_code, chunks)
        assert res.modern_code == "model_config = ConfigDict(from_attributes=True)"
        assert res.explanation == "Migrated Config"
        assert res.changes_made == ["ConfigDict"]


@pytest.mark.asyncio
async def test_synthesizer_groq_api_mock_execution() -> None:
    """Tests mocked Groq API flow to verify payload building and response parsing."""
    synthesizer = CodeSynthesizer(api_key="test-key")
    mock_response = MagicMock()
    mock_response.choices = [
        MagicMock(
            message=MagicMock(
                content=json.dumps(
                    {
                        "modern_code": "class ModernUser(BaseModel):\n    model_config = ConfigDict(from_attributes=True)",
                        "explanation": "Updated to ConfigDict",
                        "changes_made": ["Replaced Config with ConfigDict"],
                    }
                )
            )
        )
    ]
    synthesizer.client = MagicMock()
    synthesizer.client.chat.completions.create = AsyncMock(return_value=mock_response)

    chunks = [
        DocChunkPayload(
            chunk_id="chunk-1",
            topic="pydantic_config",
            breadcrumb="Pydantic > Config",
            content="model_config = ConfigDict()",
        )
    ]
    result = await synthesizer.synthesize(
        legacy_code="class Config:\n    orm_mode = True",
        context_chunks=chunks,
        feedback="Missing ConfigDict",
        iteration=2,
    )
    assert result.modern_code == "class ModernUser(BaseModel):\n    model_config = ConfigDict(from_attributes=True)"
    assert result.explanation == "Updated to ConfigDict"
    assert result.changes_made == ["Replaced Config with ConfigDict"]
    assert result.iteration == 2

    call_kwargs = synthesizer.client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "qwen/qwen3.8-27b"
    assert call_kwargs["response_format"] == {"type": "json_object"}
    assert any("Missing ConfigDict" in m["content"] for m in call_kwargs["messages"])
