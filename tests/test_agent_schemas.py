"""Unit tests for inter-agent Pydantic v2 data contracts."""

import pytest
from core_agents import (
    CriticVerdict,
    DecomposedQuery,
    DocChunkPayload,
    DraftResponse,
    GateFailure,
    VerifiedMigrationResult,
)


def test_decomposed_query_validation():
    """Tests model instantiation, whitespace stripping, and .model_dump()."""
    query = DecomposedQuery(
        summary="   Found legacy @validator and inner class Config.   ",
        detected_topics=["pydantic_validators", "pydantic_config"],
        sub_queries=["migrate @validator to @field_validator", "replace class Config with ConfigDict"],
    )

    # Whitespace stripping
    assert query.summary == "Found legacy @validator and inner class Config."
    assert query.detected_topics == ["pydantic_validators", "pydantic_config"]
    assert len(query.sub_queries) == 2

    # Model dump
    dumped = query.model_dump()
    assert isinstance(dumped, dict)
    assert dumped["summary"] == "Found legacy @validator and inner class Config."
    assert dumped["detected_topics"] == ["pydantic_validators", "pydantic_config"]
    assert dumped["sub_queries"] == [
        "migrate @validator to @field_validator",
        "replace class Config with ConfigDict",
    ]

    # Defaults check
    minimal_query = DecomposedQuery(summary="   minimal summary   ")
    assert minimal_query.summary == "minimal summary"
    assert minimal_query.detected_topics == []
    assert minimal_query.sub_queries == []


def test_doc_chunk_payload_validation():
    """Tests DocChunkPayload instantiation, whitespace stripping, and defaults."""
    payload = DocChunkPayload(
        chunk_id="  pydantic_v2_migration#field-validators  ",
        topic="pydantic_validators",
        breadcrumb="Pydantic V2 Migration > Validators",
        content="Use @field_validator instead of @validator.",
        score=0.88,
    )
    assert payload.chunk_id == "pydantic_v2_migration#field-validators"
    assert payload.topic == "pydantic_validators"
    assert payload.score == 0.88

    # Default score
    default_payload = DocChunkPayload(
        chunk_id="test_id",
        topic="test_topic",
        breadcrumb="Test > Breadcrumb",
        content="Sample content",
    )
    assert default_payload.score == 0.0


def test_draft_response_serialization():
    """Tests serialization and round-trip via .model_dump_json() and .model_validate_json()."""
    draft = DraftResponse(
        modern_code="""
from pydantic import BaseModel, ConfigDict, field_validator

class UserModel(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)
    name: str

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        return v.title()
""".strip(),
        explanation="Migrated class Config to model_config and @validator to @field_validator.",
        changes_made=[
            "Replaced class Config with model_config = ConfigDict(str_strip_whitespace=True)",
            "Replaced @validator with @field_validator and @classmethod",
        ],
        iteration=2,
    )

    json_str = draft.model_dump_json()
    assert isinstance(json_str, str)
    assert '"iteration":2' in json_str or '"iteration": 2' in json_str

    restored = DraftResponse.model_validate_json(json_str)
    assert restored == draft
    assert restored.iteration == 2
    assert len(restored.changes_made) == 2
    assert "validate_name" in restored.modern_code


def test_critic_verdict_pass_and_fail():
    """Tests passing verdict and failing verdict with GateFailure entries."""
    # 1. Passing verdict
    pass_verdict = CriticVerdict(passed=True)
    assert pass_verdict.passed is True
    assert pass_verdict.failures == []
    assert pass_verdict.suggested_fixes == []

    # 2. Failing verdict with GateFailure
    failure_1 = GateFailure(
        gate=2,
        gate_name="Negative Scan",
        message="Legacy token '@validator' found in code",
        offending_line=12,
        offending_code="  @validator('name')  ",
    )
    assert failure_1.offending_code == "@validator('name')"  # whitespace stripped

    failure_2 = GateFailure(
        gate=3,
        gate_name="Positive Scan",
        message="Missing required decorator '@classmethod' for @field_validator",
        offending_line=13,
    )
    assert failure_2.offending_code is None

    fail_verdict = CriticVerdict(
        passed=False,
        failures=[failure_1, failure_2],
        suggested_fixes=[
            "Migrate '@validator' to '@field_validator'",
            "Add '@classmethod' decorator directly below '@field_validator'",
        ],
    )

    assert fail_verdict.passed is False
    assert len(fail_verdict.failures) == 2
    assert fail_verdict.failures[0].gate == 2
    assert fail_verdict.failures[0].gate_name == "Negative Scan"
    assert fail_verdict.failures[1].gate == 3
    assert fail_verdict.failures[1].offending_code is None
    assert len(fail_verdict.suggested_fixes) == 2

    # Serialization roundtrip of CriticVerdict
    verdict_json = fail_verdict.model_dump_json()
    restored_verdict = CriticVerdict.model_validate_json(verdict_json)
    assert restored_verdict == fail_verdict


def test_verified_migration_result():
    """Tests nesting of CriticVerdict inside VerifiedMigrationResult and round-trip."""
    verdict = CriticVerdict(
        passed=True,
        failures=[],
        suggested_fixes=[],
    )

    migration_result = VerifiedMigrationResult(
        original_code="class User(BaseModel):\n    name: str\n    class Config:\n        orm_mode = True",
        modern_code="class User(BaseModel):\n    model_config = ConfigDict(from_attributes=True)\n    name: str",
        explanation="Modernized orm_mode to from_attributes in ConfigDict.",
        changes_made=["Changed orm_mode=True to from_attributes=True"],
        verdict=verdict,
        total_iterations=1,
        fallback_used=False,
    )

    assert migration_result.verdict.passed is True
    assert migration_result.total_iterations == 1
    assert migration_result.fallback_used is False

    # Serialization and nested dictionary dump
    dumped = migration_result.model_dump()
    assert isinstance(dumped["verdict"], dict)
    assert dumped["verdict"]["passed"] is True
    assert dumped["verdict"]["failures"] == []

    # JSON round trip
    json_bytes = migration_result.model_dump_json()
    restored = VerifiedMigrationResult.model_validate_json(json_bytes)
    assert restored == migration_result
    assert restored.verdict.passed is True
