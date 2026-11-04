"""Supervisor orchestrating Decomposer -> Retriever -> Synthesizer -> Critic with self-correction reflection loop."""

from __future__ import annotations

import typing

from core_agents.critic import ASTCritic
from core_agents.decomposer import QueryDecomposer
from core_agents.retriever import ScopedRetriever
from core_agents.schemas import (
    CriticVerdict,
    DecomposedQuery,
    DocChunkPayload,
    DraftResponse,
    VerifiedMigrationResult,
)


class MigrationOrchestrator:
    """Supervisor orchestrating Decomposer -> Retriever -> Synthesizer -> Critic with self-correction reflection loop."""

    def __init__(
        self,
        decomposer: typing.Optional[QueryDecomposer] = None,
        retriever: typing.Optional[ScopedRetriever] = None,
        synthesizer: typing.Optional[typing.Any] = None,
        critic: typing.Optional[ASTCritic] = None,
        max_iterations: int = 2,
    ):
        self.decomposer = decomposer or QueryDecomposer()
        self.retriever = retriever or ScopedRetriever()
        if synthesizer is None:
            from core_agents.synthesizer import CodeSynthesizer

            self.synthesizer = CodeSynthesizer()
        else:
            self.synthesizer = synthesizer
        self.critic = critic or ASTCritic()
        self.max_iterations = max_iterations

    def _verify_code(self, code: str) -> CriticVerdict:
        """Run critic verification supporting both verify and evaluate methods."""
        if hasattr(self.critic, "verify"):
            return self.critic.verify(code)
        elif hasattr(self.critic, "evaluate"):
            return self.critic.evaluate(code)
        raise AttributeError("Critic must implement 'verify' or 'evaluate'")

    @staticmethod
    def _construct_feedback(verdict: CriticVerdict) -> str:
        """Format Critic gate failures and suggested fixes into actionable prompt feedback."""
        parts: list[str] = []
        if verdict.failures:
            parts.append("Gate Failures:")
            for failure in verdict.failures:
                line_info = f" (line {failure.offending_line})" if failure.offending_line else ""
                code_info = f": `{failure.offending_code}`" if failure.offending_code else ""
                parts.append(
                    f"- Gate {failure.gate} [{failure.gate_name}]{line_info}{code_info}: {failure.message}"
                )

        if verdict.suggested_fixes:
            parts.append("Suggested Fixes:")
            for fix in verdict.suggested_fixes:
                parts.append(f"- {fix}")

        return "\n".join(parts)

    async def migrate(
        self,
        legacy_code: str,
        max_iterations: typing.Optional[int] = None,
    ) -> VerifiedMigrationResult:
        """Modernize legacy Pydantic/FastAPI code through iterative synthesis and AST critic verification."""
        # 1. Edge Case: empty or whitespace-only code
        if not legacy_code or not legacy_code.strip():
            return VerifiedMigrationResult(
                original_code=legacy_code or "",
                modern_code="",
                explanation="No code provided to modernize.",
                changes_made=[],
                verdict=CriticVerdict(passed=True, failures=[], suggested_fixes=[]),
                total_iterations=1,
                fallback_used=False,
            )

        # 2. Decompose legacy code into detected migration topics and sub-queries
        decomposed = await self.decomposer.decompose(legacy_code)

        # 3. Retrieve relevant documentation chunks
        chunks: list[DocChunkPayload] = []
        if self.retriever is not None:
            chunks = await self.retriever.retrieve(decomposed)

        # 4. Reflection Loop (up to max_iter)
        limit = max_iterations if max_iterations is not None else self.max_iterations
        limit = max(1, limit)

        feedback: typing.Optional[str] = None
        last_draft: typing.Optional[DraftResponse] = None
        last_verdict: typing.Optional[CriticVerdict] = None

        for iteration in range(1, limit + 1):
            draft = await self.synthesizer.synthesize(
                legacy_code=legacy_code,
                context_chunks=chunks,
                feedback=feedback,
                iteration=iteration,
            )
            last_draft = draft

            verdict = self._verify_code(draft.modern_code)
            last_verdict = verdict

            if verdict.passed:
                return VerifiedMigrationResult(
                    original_code=legacy_code,
                    modern_code=draft.modern_code,
                    explanation=draft.explanation,
                    changes_made=draft.changes_made,
                    verdict=verdict,
                    total_iterations=iteration,
                    fallback_used=False,
                )

            # Construct feedback with gate failures and suggested fixes for next attempt
            feedback = self._construct_feedback(verdict)

        # 5. If loop finishes without passing, return final DraftResponse with verdict
        return VerifiedMigrationResult(
            original_code=legacy_code,
            modern_code=last_draft.modern_code if last_draft else legacy_code,
            explanation=last_draft.explanation if last_draft else "Migration failed to pass verification.",
            changes_made=last_draft.changes_made if last_draft else [],
            verdict=last_verdict
            if last_verdict is not None
            else CriticVerdict(passed=False, failures=[], suggested_fixes=[]),
            total_iterations=limit,
            fallback_used=False,
        )
