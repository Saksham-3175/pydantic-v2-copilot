"""Automated evaluation test suite verifying the 6-Question Golden Benchmark."""

from __future__ import annotations

import os
from pathlib import Path
import sys
import time
import typing
import pytest
import pytest_asyncio
from qdrant_client import AsyncQdrantClient

# Ensure tests/ directory is in sys.path for relative evals package imports
tests_dir = str(Path(__file__).resolve().parent.parent)
if tests_dir not in sys.path:
    sys.path.insert(0, tests_dir)

from core_agents.critic import ASTCritic
from core_agents.decomposer import QueryDecomposer
from core_agents.orchestrator import MigrationOrchestrator
from core_agents.retriever import ScopedRetriever
from core_agents.synthesizer import CodeSynthesizer
from storage.embedder import CachedEmbedder, get_default_embedder
from storage.ingest import DEFAULT_COLLECTION_NAME, ensure_collection, ingest_corpus
from storage.qdrant_manager import QdrantStorageManager
from evals.golden_fixtures import GOLDEN_EVALS, GoldenEvalFixture
from evals.recorder import EvalRunRecorder


@pytest.fixture(scope="module")
def eval_recorder() -> typing.Generator[EvalRunRecorder, None, None]:
    """Module-scoped recorder that manages the benchmark run directory and summary."""
    recorder = EvalRunRecorder()
    recorder.start_run(tag="golden_benchmark")
    yield recorder
    summary = recorder.finish_run()
    print(f"\n[EVAL FINISHED] Run: {summary.run_id} | Pass Rate: {summary.pass_rate_percent}% ({summary.passed_evals}/{summary.total_evals})")


@pytest_asyncio.fixture
async def fully_wired_orchestrator() -> typing.AsyncGenerator[MigrationOrchestrator, None]:
    """Provide a fully wired MigrationOrchestrator backed by in-memory Qdrant and cached embeddings."""
    data_dir = Path(__file__).resolve().parent.parent.parent / "src" / "storage" / "data"
    cache_path = data_dir / "gemini_embeddings_cache.json"

    if cache_path.exists():
        embedder = CachedEmbedder(cache_path=cache_path)
    else:
        embedder = get_default_embedder()

    client = AsyncQdrantClient(":memory:")
    await ensure_collection(client, collection_name=DEFAULT_COLLECTION_NAME)
    if data_dir.exists():
        await ingest_corpus(data_dir=data_dir, client=client, embedder=embedder)

    storage_manager = QdrantStorageManager(
        client=client,
        collection_name=DEFAULT_COLLECTION_NAME,
        embedder=embedder,
    )
    retriever = ScopedRetriever(storage_manager=storage_manager)
    decomposer = QueryDecomposer()
    synthesizer = CodeSynthesizer()
    critic = ASTCritic()

    orchestrator = MigrationOrchestrator(
        decomposer=decomposer,
        retriever=retriever,
        synthesizer=synthesizer,
        critic=critic,
        max_iterations=2,
    )

    yield orchestrator
    await client.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fixture", GOLDEN_EVALS, ids=lambda f: f.eval_id)
async def test_golden_eval(
    fixture: GoldenEvalFixture,
    fully_wired_orchestrator: MigrationOrchestrator,
    eval_recorder: EvalRunRecorder,
) -> None:
    """Execute each golden eval fixture, assert all 4 gates pass, and record structured telemetry."""
    start_time = time.perf_counter()
    result = await fully_wired_orchestrator.migrate(fixture.legacy_code)
    duration_ms = (time.perf_counter() - start_time) * 1000.0

    # Gate verification breakdown
    passed_gate_1 = not any(f.gate == 1 for f in result.verdict.failures)
    passed_gate_2 = not any(f.gate == 2 for f in result.verdict.failures)
    passed_gate_3 = not any(f.gate == 3 for f in result.verdict.failures)
    passed_gate_4 = not any(f.gate == 4 for f in result.verdict.failures)

    gates_verified = {
        "gate_1_ast_syntax": passed_gate_1,
        "gate_2_negative_invariants": passed_gate_2,
        "gate_3_positive_invariants": passed_gate_3,
        "gate_4_runtime_sandbox": passed_gate_4,
    }

    # Record eval in persistent run directory
    eval_recorder.record_eval(
        eval_id=fixture.eval_id,
        topic=fixture.topic,
        passed=result.verdict.passed,
        iterations_used=result.total_iterations,
        duration_ms=duration_ms,
        legacy_input_code=fixture.legacy_code,
        modern_output_code=result.modern_code,
        critic_verdict=result.verdict.model_dump(),
        gates_verified=gates_verified,
    )

    # 1. Assert overall verdict passed
    assert result.verdict.passed is True, (
        f"Eval '{fixture.eval_id}' failed Critic Verification: {result.verdict.failures}"
    )

    # 2. Assert all forbidden legacy tokens are absent (Gate 2 Negative Invariants)
    for forbidden in fixture.forbidden_tokens:
        assert forbidden not in result.modern_code, (
            f"Forbidden legacy token '{forbidden}' leaked into modern code in {fixture.eval_id}:\n{result.modern_code}"
        )

    # 3. Assert all required modern tokens are present (Gate 3 Positive Invariants)
    for required in fixture.required_tokens:
        assert required in result.modern_code, (
            f"Required modern token '{required}' missing from modern code in {fixture.eval_id}:\n{result.modern_code}"
        )

    # 4. Assert iterations did not exceed maximum bound
    assert result.total_iterations <= 2, (
        f"Eval '{fixture.eval_id}' exceeded max iterations: {result.total_iterations}"
    )
