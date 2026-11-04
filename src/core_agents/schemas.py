"""Pydantic v2 data contracts for inter-agent communication."""

from __future__ import annotations

import typing
from pydantic import BaseModel, ConfigDict, Field


class DecomposedQuery(BaseModel):
    """Output from the Query Decomposer agent."""

    model_config = ConfigDict(str_strip_whitespace=True)

    summary: str = Field(description="Summary of detected legacy patterns in user code")
    detected_topics: list[str] = Field(
        default_factory=list,
        description="List of detected topic taxonomies (e.g. 'pydantic_validators', 'pydantic_config', 'fastapi_lifespan')",
    )
    sub_queries: list[str] = Field(
        default_factory=list,
        description="Targeted search query strings for vector retrieval",
    )


class DocChunkPayload(BaseModel):
    """Normalized document chunk payload passed to the Synthesizer."""

    model_config = ConfigDict(str_strip_whitespace=True)

    chunk_id: str
    topic: str
    breadcrumb: str
    content: str
    score: float = 0.0


class DraftResponse(BaseModel):
    """Candidate modernization produced by the Synthesizer agent."""

    model_config = ConfigDict(str_strip_whitespace=True)

    modern_code: str = Field(description="Modernized Python code block")
    explanation: str = Field(description="Explanation of architectural changes made")
    changes_made: list[str] = Field(
        default_factory=list,
        description="Bullet items of transformations",
    )
    iteration: int = Field(
        default=1,
        description="Attempt number in the supervisor reflection loop",
    )


class GateFailure(BaseModel):
    """Details of a single verification gate failure reported by the AST Critic."""

    model_config = ConfigDict(str_strip_whitespace=True)

    gate: int = Field(description="Gate number: 1 (AST Parse), 2 (Negative Scan), 3 (Positive Scan), 4 (Runtime)")
    gate_name: str = Field(description="Name of the failed gate")
    message: str = Field(description="Descriptive failure reason")
    offending_line: typing.Optional[int] = None
    offending_code: typing.Optional[str] = None


class CriticVerdict(BaseModel):
    """Deterministic verdict output from the AST & Rule Critic."""

    model_config = ConfigDict(str_strip_whitespace=True)

    passed: bool = Field(description="True if all 4 gates passed without failure")
    failures: list[GateFailure] = Field(default_factory=list)
    suggested_fixes: list[str] = Field(default_factory=list)


class VerifiedMigrationResult(BaseModel):
    """Final output returned to the API gateway and user."""

    model_config = ConfigDict(str_strip_whitespace=True)

    original_code: str
    modern_code: str
    explanation: str
    changes_made: list[str] = Field(default_factory=list)
    verdict: CriticVerdict
    total_iterations: int = 1
    fallback_used: bool = False
