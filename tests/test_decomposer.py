"""Unit tests for QueryDecomposer agent covering happy paths and edge cases."""

from unittest.mock import AsyncMock, MagicMock
import pytest
from core_agents.decomposer import QueryDecomposer
from core_agents.schemas import DecomposedQuery


@pytest.fixture
def offline_decomposer(monkeypatch: pytest.MonkeyPatch) -> QueryDecomposer:
    """Fixture returning QueryDecomposer in deterministic offline mode."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    return QueryDecomposer(api_key=None)


@pytest.mark.asyncio
async def test_decomposer_happy_path_validators_and_config(
    offline_decomposer: QueryDecomposer,
) -> None:
    """Feeds code with @validator and class Config:, asserts detected_topics has both."""
    code = """
from pydantic import BaseModel, validator

class UserModel(BaseModel):
    name: str
    age: int

    class Config:
        orm_mode = True

    @validator("name")
    def name_must_be_capitalized(cls, v: str) -> str:
        return v.title()
"""
    result = await offline_decomposer.decompose(code)
    assert isinstance(result, DecomposedQuery)
    assert "pydantic_validators" in result.detected_topics
    assert "pydantic_config" in result.detected_topics
    assert len(result.sub_queries) == 2
    assert any("field_validator" in q for q in result.sub_queries)
    assert any("ConfigDict" in q for q in result.sub_queries)
    assert "Detected 2 legacy patterns" in result.summary


@pytest.mark.asyncio
async def test_decomposer_edge_case_empty_string(
    offline_decomposer: QueryDecomposer,
) -> None:
    """Feeds empty string and whitespace-only string, asserts topics & sub-queries are empty."""
    empty_result = await offline_decomposer.decompose("")
    assert empty_result.detected_topics == []
    assert empty_result.sub_queries == []
    assert empty_result.summary == "No code provided."

    whitespace_result = await offline_decomposer.decompose("   \n\t  \n  ")
    assert whitespace_result.detected_topics == []
    assert whitespace_result.sub_queries == []
    assert whitespace_result.summary == "No code provided."


@pytest.mark.asyncio
async def test_decomposer_edge_case_already_modern_code(
    offline_decomposer: QueryDecomposer,
) -> None:
    """Feeds modern Pydantic v2 code (@field_validator, ConfigDict), asserts detected_topics == []."""
    modern_code = """
from pydantic import BaseModel, ConfigDict, field_validator

class ModernModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return v.strip()
"""
    result = await offline_decomposer.decompose(modern_code)
    assert result.detected_topics == []
    assert result.sub_queries == []
    assert "No legacy Pydantic v1 or FastAPI patterns detected" in result.summary


@pytest.mark.asyncio
async def test_decomposer_edge_case_comments_only(
    offline_decomposer: QueryDecomposer,
) -> None:
    """Feeds comments only, asserts detected_topics == []."""
    comment_code = """
# just a comment explaining nothing
# no legacy pydantic constructs here
"""
    result = await offline_decomposer.decompose(comment_code)
    assert result.detected_topics == []
    assert result.sub_queries == []
    assert "No legacy Pydantic v1 or FastAPI patterns detected" in result.summary


@pytest.mark.asyncio
async def test_decomposer_all_topics(
    offline_decomposer: QueryDecomposer,
) -> None:
    """Feeds code containing all 6 legacy patterns; asserts all 6 topics detected."""
    all_legacy_code = """
from pydantic import BaseModel, BaseSettings, validator, root_validator
from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
def startup():
    pass

class Settings(BaseSettings):
    db_url: str

class OldModel(BaseModel):
    val: str

    class Config:
        orm_mode = True

    @validator("val")
    def check_val(cls, v):
        return v

    @root_validator
    def check_all(cls, values):
        return values

m = OldModel(val="test")
data = m.dict()
json_str = m.json()
"""
    result = await offline_decomposer.decompose(all_legacy_code)
    assert len(result.detected_topics) == 6
    expected_topics = [
        "pydantic_validators",
        "pydantic_model_validators",
        "pydantic_config",
        "pydantic_serialization",
        "pydantic_settings",
        "fastapi_lifespan",
    ]
    for topic in expected_topics:
        assert topic in result.detected_topics

    assert len(result.sub_queries) == 6
    assert "Detected 6 legacy patterns" in result.summary


@pytest.mark.asyncio
async def test_decomposer_gemini_api_success(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests the structured output response handling when Gemini API client succeeds."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    decomposer = QueryDecomposer(api_key="fake-test-key")

    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.text = '{"summary": "Mocked analysis", "detected_topics": ["pydantic_config"], "sub_queries": ["mocked query"]}'
    mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
    decomposer.client = mock_client

    result = await decomposer.decompose("class Config:\n    orm_mode = True")
    assert result.summary == "Mocked analysis"
    assert result.detected_topics == ["pydantic_config"]
    assert result.sub_queries == ["mocked query"]
    mock_client.aio.models.generate_content.assert_awaited_once()


@pytest.mark.asyncio
async def test_decomposer_gemini_api_fallback_on_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tests fallback to heuristic scanner when live API call fails."""
    monkeypatch.setenv("GEMINI_API_KEY", "fake-test-key")
    decomposer = QueryDecomposer(api_key="fake-test-key")

    mock_client = MagicMock()
    mock_client.aio.models.generate_content = AsyncMock(side_effect=RuntimeError("API error"))
    decomposer.client = mock_client

    result = await decomposer.decompose("@validator('x')\ndef val(cls, v): return v")
    assert "pydantic_validators" in result.detected_topics
    assert len(result.sub_queries) == 1
    assert "how to migrate @validator" in result.sub_queries[0]
