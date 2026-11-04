"""Integration and contract tests for FastAPI migration endpoints and SSE streaming."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch
import httpx
import pytest
import pytest_asyncio

from copilot_api.main import app, lifespan
from copilot_api.routes import get_orchestrator
from core_agents.orchestrator import MigrationOrchestrator
from core_agents.schemas import CriticVerdict, VerifiedMigrationResult


@pytest_asyncio.fixture
async def client():
    """Async client fixture entering app lifespan for full lifecycle management."""
    async with lifespan(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as ac:
            yield ac


@pytest.mark.asyncio
async def test_health_check(client: httpx.AsyncClient) -> None:
    """GET /health returns 200 and healthy status metadata."""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "pydantic-v2-copilot"
    assert data["version"] == "0.1.0"


@pytest.mark.asyncio
async def test_migrate_legacy_validator(client: httpx.AsyncClient) -> None:
    """POST /v1/migrate modernizes legacy @validator code and returns verified result."""
    legacy_code = """
from pydantic import BaseModel, validator

class UserModel(BaseModel):
    name: str

    @validator("name")
    def validate_name(cls, v):
        return v.title()
"""
    response = await client.post(
        "/v1/migrate",
        json={"code": legacy_code, "max_iterations": 2},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["verdict"]["passed"] is True
    assert len(data["verdict"]["failures"]) == 0
    assert "@field_validator" in data["modern_code"]
    assert "@classmethod" in data["modern_code"]
    assert "@validator(" not in data["modern_code"]
    assert data["total_iterations"] >= 1


@pytest.mark.asyncio
async def test_migrate_legacy_config(client: httpx.AsyncClient) -> None:
    """POST /v1/migrate modernizes legacy inner class Config to ConfigDict."""
    legacy_code = """
from pydantic import BaseModel

class Settings(BaseModel):
    name: str

    class Config:
        str_strip_whitespace = True
"""
    response = await client.post(
        "/v1/migrate",
        json={"code": legacy_code, "max_iterations": 2},
    )
    assert response.status_code == 200
    data = response.json()

    assert data["verdict"]["passed"] is True
    assert "ConfigDict" in data["modern_code"]
    assert "class Config:" not in data["modern_code"]


@pytest.mark.asyncio
async def test_migrate_empty_string(client: httpx.AsyncClient) -> None:
    """POST /v1/migrate with empty or whitespace string returns 200 and clean result."""
    for empty_input in ("", "   ", "\n\t  \n"):
        response = await client.post(
            "/v1/migrate",
            json={"code": empty_input},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["verdict"]["passed"] is True
        assert data["modern_code"] == ""
        assert data["changes_made"] == []
        assert data["total_iterations"] == 1


@pytest.mark.asyncio
async def test_migrate_stream_progress_events(client: httpx.AsyncClient) -> None:
    """POST /v1/migrate/stream yields SSE events through all stages leading to 'done'."""
    legacy_code = """
from pydantic import BaseModel, validator

class Item(BaseModel):
    name: str

    @validator("name")
    def check_name(cls, v):
        return v.strip()
"""
    async with client.stream(
        "POST",
        "/v1/migrate/stream",
        json={"code": legacy_code, "max_iterations": 2},
    ) as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        assert response.headers.get("cache-control") == "no-cache"

        events: list[dict] = []
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                payload = json.loads(line.removeprefix("data: "))
                events.append(payload)

    stages = [e["stage"] for e in events]
    assert "start" in stages
    assert "decomposed" in stages
    assert "retrieved" in stages
    assert "synthesizing" in stages
    assert "critic_verification" in stages
    assert "done" in stages

    done_event = next(e for e in events if e["stage"] == "done")
    result = done_event["result"]
    assert result["verdict"]["passed"] is True
    assert "@field_validator" in result["modern_code"]


@pytest.mark.asyncio
async def test_migrate_stream_empty_code(client: httpx.AsyncClient) -> None:
    """POST /v1/migrate/stream with empty code yields start and immediately done."""
    async with client.stream(
        "POST",
        "/v1/migrate/stream",
        json={"code": "   "},
    ) as response:
        assert response.status_code == 200
        events: list[dict] = []
        async for line in response.aiter_lines():
            if line.startswith("data: "):
                payload = json.loads(line.removeprefix("data: "))
                events.append(payload)

    stages = [e["stage"] for e in events]
    assert stages == ["start", "done"]
    done_event = events[1]
    assert done_event["result"]["modern_code"] == ""
    assert done_event["result"]["verdict"]["passed"] is True


@pytest.mark.asyncio
async def test_migrate_stream_client_disconnect(client: httpx.AsyncClient) -> None:
    """POST /v1/migrate/stream stops generation when client disconnect is detected."""
    call_count = 0

    async def fake_is_disconnected() -> bool:
        nonlocal call_count
        call_count += 1
        # Disconnect right after yielding 'start'
        return call_count > 1

    with patch("starlette.requests.Request.is_disconnected", side_effect=fake_is_disconnected):
        legacy_code = """
from pydantic import BaseModel, validator

class Item(BaseModel):
    name: str

    @validator("name")
    def check_name(cls, v):
        return v
"""
        events: list[dict] = []
        async with client.stream(
            "POST",
            "/v1/migrate/stream",
            json={"code": legacy_code},
        ) as response:
            assert response.status_code == 200
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    events.append(json.loads(line.removeprefix("data: ")))

        stages = [e["stage"] for e in events]
        assert "start" in stages
        # Prevented downstream expensive stages
        assert "synthesizing" not in stages
        assert "critic_verification" not in stages
        assert "done" not in stages


@pytest.mark.asyncio
async def test_migrate_validation_error(client: httpx.AsyncClient) -> None:
    """POST /v1/migrate with out-of-range max_iterations returns 422 Unprocessable Entity."""
    response = await client.post(
        "/v1/migrate",
        json={"code": "name: str", "max_iterations": 10},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_dependency_injection_override(client: httpx.AsyncClient) -> None:
    """Verifies get_orchestrator can be overridden for testing via dependency_overrides."""
    mock_orchestrator = AsyncMock(spec=MigrationOrchestrator)
    mock_orchestrator.migrate.return_value = VerifiedMigrationResult(
        original_code="original",
        modern_code="modern_mocked",
        explanation="Mocked modernization explanation",
        changes_made=["Mocked change"],
        verdict=CriticVerdict(passed=True, failures=[], suggested_fixes=[]),
        total_iterations=1,
        fallback_used=False,
    )

    app.dependency_overrides[get_orchestrator] = lambda: mock_orchestrator
    try:
        response = await client.post(
            "/v1/migrate",
            json={"code": "original"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["modern_code"] == "modern_mocked"
        assert mock_orchestrator.migrate.await_count == 1
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_root_prefix_routes(client: httpx.AsyncClient) -> None:
    """POST /migrate works with or without /v1 prefix."""
    response = await client.post("/migrate", json={"code": ""})
    assert response.status_code == 200
    assert response.json()["verdict"]["passed"] is True
