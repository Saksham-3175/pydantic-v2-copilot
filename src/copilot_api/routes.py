from __future__ import annotations

import json
import typing
from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from core_agents.orchestrator import MigrationOrchestrator
from core_agents.schemas import CriticVerdict, VerifiedMigrationResult

router = APIRouter()


class MigrationRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True)

    code: str = Field(..., description="Legacy Python code snippet to modernize")
    max_iterations: int = Field(
        default=2,
        ge=1,
        le=5,
        description="Maximum self-correction attempts",
    )


def get_orchestrator(request: Request) -> MigrationOrchestrator:
    if hasattr(request.app.state, "orchestrator") and request.app.state.orchestrator is not None:
        return request.app.state.orchestrator
    orchestrator = MigrationOrchestrator()
    request.app.state.orchestrator = orchestrator
    return orchestrator


@router.post("/migrate", response_model=VerifiedMigrationResult)
async def migrate_code(
    payload: MigrationRequest,
    orchestrator: typing.Annotated[MigrationOrchestrator, Depends(get_orchestrator)],
) -> VerifiedMigrationResult:
    return await orchestrator.migrate(
        legacy_code=payload.code,
        max_iterations=payload.max_iterations,
    )


@router.post("/migrate/stream")
async def migrate_code_stream(
    request: Request,
    payload: MigrationRequest,
    orchestrator: typing.Annotated[MigrationOrchestrator, Depends(get_orchestrator)],
) -> StreamingResponse:
    async def event_generator():
        # Check disconnect: if await request.is_disconnected(): return
        if await request.is_disconnected():
            return

        # 1. start
        start_event = {
            "stage": "start",
            "message": "Starting migration analysis and modernization",
        }
        yield f"data: {json.dumps(start_event)}\n\n"

        legacy_code = payload.code

        # Edge case: empty or whitespace-only code
        if not legacy_code or not legacy_code.strip():
            empty_result = VerifiedMigrationResult(
                original_code=legacy_code or "",
                modern_code="",
                explanation="No code provided to modernize.",
                changes_made=[],
                verdict=CriticVerdict(passed=True, failures=[], suggested_fixes=[]),
                total_iterations=1,
                fallback_used=False,
            )
            done_event = {
                "stage": "done",
                "result": empty_result.model_dump(),
                "modern_code": empty_result.modern_code,
                "passed": empty_result.verdict.passed,
            }
            yield f"data: {json.dumps(done_event)}\n\n"
            return

        if await request.is_disconnected():
            return

        # 2. decomposed
        decomposed = await orchestrator.decomposer.decompose(legacy_code)
        decomposed_event = {
            "stage": "decomposed",
            "summary": decomposed.summary,
            "detected_topics": decomposed.detected_topics,
            "sub_queries": decomposed.sub_queries,
            "data": decomposed.model_dump(),
        }
        yield f"data: {json.dumps(decomposed_event)}\n\n"

        if await request.is_disconnected():
            return

        # 3. retrieved
        chunks = []
        if orchestrator.retriever is not None:
            chunks = await orchestrator.retriever.retrieve(decomposed)
        retrieved_event = {
            "stage": "retrieved",
            "count": len(chunks),
            "chunks": [c.model_dump() for c in chunks],
        }
        yield f"data: {json.dumps(retrieved_event)}\n\n"

        # 4. Reflection Loop
        limit = payload.max_iterations if payload.max_iterations is not None else orchestrator.max_iterations
        limit = max(1, limit)

        feedback: typing.Optional[str] = None
        last_draft = None
        last_verdict = None

        for iteration in range(1, limit + 1):
            if await request.is_disconnected():
                return

            # Synthesizing
            synthesizing_event = {
                "stage": "synthesizing",
                "iteration": iteration,
            }
            yield f"data: {json.dumps(synthesizing_event)}\n\n"

            draft = await orchestrator.synthesizer.synthesize(
                legacy_code=legacy_code,
                context_chunks=chunks,
                feedback=feedback,
                iteration=iteration,
            )
            last_draft = draft

            if await request.is_disconnected():
                return

            # Critic verification
            verdict = orchestrator._verify_code(draft.modern_code)
            last_verdict = verdict

            critic_event = {
                "stage": "critic_verification",
                "iteration": iteration,
                "passed": verdict.passed,
                "verdict": verdict.model_dump(),
            }
            yield f"data: {json.dumps(critic_event)}\n\n"

            if verdict.passed:
                final_result = VerifiedMigrationResult(
                    original_code=legacy_code,
                    modern_code=draft.modern_code,
                    explanation=draft.explanation,
                    changes_made=draft.changes_made,
                    verdict=verdict,
                    total_iterations=iteration,
                    fallback_used=False,
                )
                done_event = {
                    "stage": "done",
                    "result": final_result.model_dump(),
                    "modern_code": final_result.modern_code,
                    "passed": final_result.verdict.passed,
                }
                yield f"data: {json.dumps(done_event)}\n\n"
                return

            feedback = orchestrator._construct_feedback(verdict)

        # 5. Reflection loop exhausted without passing
        final_result = VerifiedMigrationResult(
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
        done_event = {
            "stage": "done",
            "result": final_result.model_dump(),
            "modern_code": final_result.modern_code,
            "passed": final_result.verdict.passed,
        }
        yield f"data: {json.dumps(done_event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
