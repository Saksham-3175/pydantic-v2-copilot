"""Unit tests for MigrationOrchestrator supervisor agent."""

from unittest.mock import AsyncMock, MagicMock
import pytest

from core_agents.critic import ASTCritic
from core_agents.decomposer import QueryDecomposer
from core_agents.orchestrator import MigrationOrchestrator
from core_agents.retriever import ScopedRetriever
from core_agents.schemas import (
    CriticVerdict,
    DecomposedQuery,
    DocChunkPayload,
    DraftResponse,
    GateFailure,
    VerifiedMigrationResult,
)
from core_agents.synthesizer import CodeSynthesizer


@pytest.mark.asyncio
async def test_orchestrator_happy_path_single_iteration() -> None:
    """Mock or real components pass on iteration 1; asserts passed == True and total_iterations == 1."""
    mock_decomposer = AsyncMock()
    mock_decomposer.decompose.return_value = DecomposedQuery(
        summary="Legacy validator and config",
        detected_topics=["pydantic_validators", "pydantic_config"],
        sub_queries=["migrate validator to field_validator", "replace class Config with ConfigDict"],
    )

    mock_retriever = AsyncMock()
    mock_retriever.retrieve.return_value = [
        DocChunkPayload(
            chunk_id="pydantic_v2_migration#field-validation",
            topic="pydantic_validators",
            breadcrumb="Validators > Field Validator",
            content="Use @field_validator with @classmethod",
        )
    ]

    mock_synthesizer = AsyncMock()
    modern_valid_code = (
        "from pydantic import BaseModel, ConfigDict\n\n"
        "class UserModel(BaseModel):\n"
        "    model_config = ConfigDict(str_strip_whitespace=True)\n"
        "    name: str\n"
    )
    mock_synthesizer.synthesize.return_value = DraftResponse(
        modern_code=modern_valid_code,
        explanation="Modernized to Pydantic v2 ConfigDict.",
        changes_made=["Replaced Config with ConfigDict"],
        iteration=1,
    )

    critic = ASTCritic()

    orchestrator = MigrationOrchestrator(
        decomposer=mock_decomposer,
        retriever=mock_retriever,
        synthesizer=mock_synthesizer,
        critic=critic,
        max_iterations=2,
    )

    result = await orchestrator.migrate(legacy_code="class Config:\n    str_strip_whitespace = True")

    assert isinstance(result, VerifiedMigrationResult)
    assert result.verdict.passed is True
    assert result.total_iterations == 1
    assert result.modern_code == modern_valid_code.strip()
    assert result.fallback_used is False
    assert len(result.changes_made) == 1
    assert mock_decomposer.decompose.await_count == 1
    assert mock_retriever.retrieve.await_count == 1
    assert mock_synthesizer.synthesize.await_count == 1


@pytest.mark.asyncio
async def test_orchestrator_reflection_retry_loop() -> None:
    """Mock synthesizer produces failing code on iteration 1 (.dict()), but with feedback produces valid code on iteration 2 (.model_dump())."""
    mock_decomposer = AsyncMock()
    mock_decomposer.decompose.return_value = DecomposedQuery(
        summary="Legacy dict serialization",
        detected_topics=["pydantic_serialization"],
        sub_queries=["migrate dict to model_dump"],
    )

    mock_retriever = AsyncMock()
    mock_retriever.retrieve.return_value = []

    mock_synthesizer = AsyncMock()

    failing_code = (
        "from pydantic import BaseModel\n\n"
        "class UserModel(BaseModel):\n"
        "    name: str\n\n"
        "user = UserModel(name='Alice')\n"
        "data = user.dict()\n"
    )

    passing_code = (
        "from pydantic import BaseModel\n\n"
        "class UserModel(BaseModel):\n"
        "    name: str\n\n"
        "user = UserModel(name='Alice')\n"
        "data = user.model_dump()\n"
    )

    async def fake_synthesize(
        legacy_code: str,
        context_chunks: list[DocChunkPayload],
        feedback: str | None = None,
        iteration: int = 1,
    ) -> DraftResponse:
        if iteration == 1:
            assert feedback is None
            return DraftResponse(
                modern_code=failing_code,
                explanation="Draft 1 still contains .dict()",
                changes_made=[],
                iteration=1,
            )
        else:
            assert feedback is not None
            assert ".dict()" in feedback or "dict" in feedback.lower()
            return DraftResponse(
                modern_code=passing_code,
                explanation="Draft 2 fixed with .model_dump()",
                changes_made=["Replaced .dict() with .model_dump()"],
                iteration=2,
            )

    mock_synthesizer.synthesize.side_effect = fake_synthesize

    critic = ASTCritic()

    orchestrator = MigrationOrchestrator(
        decomposer=mock_decomposer,
        retriever=mock_retriever,
        synthesizer=mock_synthesizer,
        critic=critic,
        max_iterations=3,
    )

    result = await orchestrator.migrate(legacy_code="data = user.dict()")

    assert result.verdict.passed is True
    assert result.total_iterations == 2
    assert "model_dump" in result.modern_code
    assert mock_synthesizer.synthesize.await_count == 2


@pytest.mark.asyncio
async def test_orchestrator_max_iterations_exhausted() -> None:
    """Mock synthesizer always produces failing code; terminates at max_iterations with passed == False."""
    mock_decomposer = AsyncMock()
    mock_decomposer.decompose.return_value = DecomposedQuery(
        summary="Legacy validator",
        detected_topics=["pydantic_validators"],
        sub_queries=["migrate validator"],
    )

    mock_retriever = AsyncMock()
    mock_retriever.retrieve.return_value = []

    mock_synthesizer = AsyncMock()
    broken_code = (
        "from pydantic import BaseModel, validator\n\n"
        "class Item(BaseModel):\n"
        "    name: str\n\n"
        "    @validator('name')\n"
        "    def check_name(cls, v):\n"
        "        return v\n"
    )

    mock_synthesizer.synthesize.return_value = DraftResponse(
        modern_code=broken_code,
        explanation="Draft still contains @validator",
        changes_made=[],
        iteration=1,
    )

    critic = ASTCritic()

    orchestrator = MigrationOrchestrator(
        decomposer=mock_decomposer,
        retriever=mock_retriever,
        synthesizer=mock_synthesizer,
        critic=critic,
        max_iterations=2,
    )

    result = await orchestrator.migrate(legacy_code="class Item(BaseModel): pass", max_iterations=2)

    assert result.verdict.passed is False
    assert result.total_iterations == 2
    assert len(result.verdict.failures) > 0
    assert any("validator" in f.message for f in result.verdict.failures)
    assert mock_synthesizer.synthesize.await_count == 2


@pytest.mark.asyncio
async def test_orchestrator_edge_case_empty_code() -> None:
    """Passes empty string and whitespace; immediately returns passed == True without calling retriever or synthesizer."""
    mock_decomposer = AsyncMock()
    mock_retriever = AsyncMock()
    mock_synthesizer = AsyncMock()
    critic = ASTCritic()

    orchestrator = MigrationOrchestrator(
        decomposer=mock_decomposer,
        retriever=mock_retriever,
        synthesizer=mock_synthesizer,
        critic=critic,
    )

    for empty_input in ("", "   ", "\n\t  \n  "):
        result = await orchestrator.migrate(legacy_code=empty_input)
        assert result.verdict.passed is True
        assert result.total_iterations == 1
        assert result.modern_code == ""
        assert result.changes_made == []
        assert result.explanation == "No code provided to modernize."

    assert mock_decomposer.decompose.await_count == 0
    assert mock_retriever.retrieve.await_count == 0
    assert mock_synthesizer.synthesize.await_count == 0


@pytest.mark.asyncio
async def test_orchestrator_end_to_end_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Modernizes legacy code with @validator and class Config: using offline components, asserting result passes AST Critic."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    decomposer = QueryDecomposer(api_key=None)
    retriever = ScopedRetriever(storage_manager=None)
    synthesizer = CodeSynthesizer(api_key=None)
    critic = ASTCritic()

    orchestrator = MigrationOrchestrator(
        decomposer=decomposer,
        retriever=retriever,
        synthesizer=synthesizer,
        critic=critic,
        max_iterations=2,
    )

    legacy_code = """
from pydantic import BaseModel, validator

class UserModel(BaseModel):
    name: str

    class Config:
        str_strip_whitespace = True

    @validator("name")
    def name_must_be_capitalized(cls, v: str) -> str:
        return v.title()
"""

    result = await orchestrator.migrate(legacy_code=legacy_code)

    assert isinstance(result, VerifiedMigrationResult)
    assert result.verdict.passed is True
    assert len(result.verdict.failures) == 0
    assert result.total_iterations >= 1
    assert "ConfigDict" in result.modern_code
    assert "@field_validator" in result.modern_code
    assert "@classmethod" in result.modern_code
    assert "class Config:" not in result.modern_code
    assert "@validator(" not in result.modern_code


def test_orchestrator_default_instantiation() -> None:
    """Verifies MigrationOrchestrator can be initialized with default arguments."""
    orchestrator = MigrationOrchestrator()
    assert isinstance(orchestrator.decomposer, QueryDecomposer)
    assert isinstance(orchestrator.retriever, ScopedRetriever)
    assert isinstance(orchestrator.critic, ASTCritic)
    assert orchestrator.max_iterations == 2


def test_orchestrator_feedback_formatting() -> None:
    """Verifies structured feedback generation from gate failures and suggested fixes."""
    failures = [
        GateFailure(
            gate=2,
            gate_name="Negative Invariant Scan",
            message="Legacy .dict() method is deprecated.",
            offending_line=12,
            offending_code="user.dict()",
        )
    ]
    verdict = CriticVerdict(
        passed=False,
        failures=failures,
        suggested_fixes=["Replace .dict() with .model_dump()"],
    )

    feedback = MigrationOrchestrator._construct_feedback(verdict)
    assert "Gate 2 [Negative Invariant Scan]" in feedback
    assert "(line 12)" in feedback
    assert "`user.dict()`" in feedback
    assert "Suggested Fixes:" in feedback
    assert "Replace .dict() with .model_dump()" in feedback
