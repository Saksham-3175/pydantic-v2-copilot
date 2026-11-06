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
        async for event in orchestrator.migrate_stream(
            legacy_code=payload.code,
            max_iterations=payload.max_iterations,
            is_cancelled=request.is_disconnected,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
