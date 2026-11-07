"""Unit and integration tests for copilot CLI commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from copilot_cli.main import app

runner = CliRunner()


def test_cli_version() -> None:
    """Verifies copilot version displays architectural info and LLM configuration."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout
    assert "Deterministic Critic" in result.stdout
    assert "Decomposer" in result.stdout


def test_cli_root_version_flags() -> None:
    """Verifies copilot --version and copilot -v display architectural info and LLM configuration."""
    res_long = runner.invoke(app, ["--version"])
    assert res_long.exit_code == 0
    assert "0.1.0" in res_long.stdout
    assert "Deterministic Critic" in res_long.stdout

    res_short = runner.invoke(app, ["-v"])
    assert res_short.exit_code == 0
    assert "0.1.0" in res_short.stdout
    assert "Deterministic Critic" in res_short.stdout


def test_cli_scan_clean_directory(tmp_path: Path) -> None:
    """Verifies copilot scan reports clean files without legacy tokens."""
    clean_file = tmp_path / "utils.py"
    clean_file.write_text(
        "def add(a: int, b: int) -> int:\n    return a + b\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["scan", str(tmp_path)])
    assert result.exit_code == 0
    assert "CLEAN" in result.stdout
    assert "utils.py" in result.stdout


def test_cli_scan_legacy_directory(tmp_path: Path) -> None:
    """Verifies copilot scan flags legacy Pydantic v1 patterns."""
    legacy_file = tmp_path / "model.py"
    legacy_file.write_text(
        "from pydantic import BaseModel\n\n"
        "class Item(BaseModel):\n"
        "    class Config:\n"
        "        orm_mode = True\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["scan", str(tmp_path)])
    assert result.exit_code == 0
    assert "NEEDS MIGRATION" in result.stdout
    assert "model.py" in result.stdout


def test_cli_migrate_dry_run_preserves_disk(tmp_path: Path) -> None:
    """Verifies copilot migrate --diff (or --dry-run) does not mutate files on disk."""
    legacy_code = (
        "from pydantic import BaseModel\n\n"
        "class Item(BaseModel):\n"
        "    class Config:\n"
        "        orm_mode = True\n"
    )
    modern_code = (
        "from pydantic import BaseModel, ConfigDict\n\n"
        "class Item(BaseModel):\n"
        "    model_config = ConfigDict(from_attributes=True)\n"
    )

    target_file = tmp_path / "target.py"
    target_file.write_text(legacy_code, encoding="utf-8")

    async def fake_stream(*args, **kwargs):
        yield {"stage": "start"}
        yield {"stage": "decomposed", "detected_topics": ["pydantic_config"]}
        yield {"stage": "retrieved", "count": 1}
        yield {"stage": "synthesizing", "iteration": 1}
        yield {"stage": "critic_verification", "iteration": 1, "passed": True}
        yield {
            "stage": "done",
            "passed": True,
            "modern_code": modern_code,
            "result": {"total_iterations": 1},
        }

    with patch("copilot_cli.main.MigrationOrchestrator.migrate_stream", side_effect=fake_stream):
        result = runner.invoke(app, ["migrate", str(target_file), "--diff"])
        assert result.exit_code == 0
        assert "Diff:" in result.stdout
        assert "DRY-RUN" in result.stdout
        # Verify file on disk was NOT mutated
        assert target_file.read_text(encoding="utf-8") == legacy_code


def test_cli_migrate_write_in_place_and_skips_clean(tmp_path: Path) -> None:
    """Verifies copilot migrate --write modernizes flagged file and leaves clean file untouched."""
    legacy_code = (
        "from pydantic import BaseModel\n\n"
        "class User(BaseModel):\n"
        "    class Config:\n"
        "        orm_mode = True\n"
    )
    modern_code = (
        "from pydantic import BaseModel, ConfigDict\n\n"
        "class User(BaseModel):\n"
        "    model_config = ConfigDict(from_attributes=True)\n"
    )
    clean_code = "def multiply(x: int, y: int) -> int:\n    return x * y\n"

    target_file = tmp_path / "user.py"
    clean_file = tmp_path / "math_utils.py"
    target_file.write_text(legacy_code, encoding="utf-8")
    clean_file.write_text(clean_code, encoding="utf-8")

    async def fake_stream(*args, **kwargs):
        yield {"stage": "start"}
        yield {"stage": "synthesizing", "iteration": 1}
        yield {"stage": "critic_verification", "iteration": 1, "passed": True}
        yield {
            "stage": "done",
            "passed": True,
            "modern_code": modern_code,
            "result": {"total_iterations": 1},
        }

    with patch("copilot_cli.main.MigrationOrchestrator.migrate_stream", side_effect=fake_stream):
        result = runner.invoke(app, ["migrate", str(tmp_path), "--write"])
        assert result.exit_code == 0
        assert "MIGRATED" in result.stdout
        assert "SKIPPED" in result.stdout
        # Target file must be updated in-place
        assert target_file.read_text(encoding="utf-8") == modern_code
        # Clean file must remain identical
        assert clean_file.read_text(encoding="utf-8") == clean_code


def test_cli_migrate_run_tests_success(tmp_path: Path) -> None:
    """Verifies --run-tests invokes pytest runner and succeeds."""
    clean_file = tmp_path / "helpers.py"
    clean_file.write_text("def salute() -> str:\n    return 'hello'\n", encoding="utf-8")

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_sample.py"
    test_file.write_text(
        "from helpers import salute\n\ndef test_salute():\n    assert salute() == 'hello'\n",
        encoding="utf-8",
    )

    result = runner.invoke(app, ["migrate", str(tmp_path), "--write", "--run-tests"])
    assert result.exit_code == 0
    assert "Target test suite passed with 100% success!" in result.stdout


def test_cli_migrate_cache_avoids_duplicate_calls(tmp_path: Path) -> None:
    """Verifies that running --diff caches the result and --write uses the cache without re-invoking the orchestrator."""
    from copilot_cli.cache import MigrationCache

    legacy_code = (
        "from pydantic import BaseModel\n\n"
        "class Model(BaseModel):\n"
        "    class Config:\n"
        "        orm_mode = True\n"
    )
    modern_code = (
        "from pydantic import BaseModel, ConfigDict\n\n"
        "class Model(BaseModel):\n"
        "    model_config = ConfigDict(from_attributes=True)\n"
    )

    target_file = tmp_path / "model.py"
    target_file.write_text(legacy_code, encoding="utf-8")
    test_cache_file = tmp_path / "test_cache.json"

    call_count = 0

    async def counting_stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        yield {"stage": "start"}
        yield {"stage": "synthesizing", "iteration": 1}
        yield {"stage": "critic_verification", "iteration": 1, "passed": True}
        yield {
            "stage": "done",
            "passed": True,
            "modern_code": modern_code,
            "result": {"total_iterations": 1},
        }

    with patch("copilot_cli.main.MigrationCache", return_value=MigrationCache(cache_file=test_cache_file)), \
         patch("copilot_cli.main.MigrationOrchestrator.migrate_stream", side_effect=counting_stream):

        # 1. First run with --diff: must call orchestrator and cache the result
        res1 = runner.invoke(app, ["migrate", str(target_file), "--diff"])
        assert res1.exit_code == 0
        assert call_count == 1
        assert "DRY-RUN" in res1.stdout
        # File is untouched
        assert target_file.read_text(encoding="utf-8") == legacy_code

        # 2. Second run with --write: must USE CACHE and NOT call orchestrator
        res2 = runner.invoke(app, ["migrate", str(target_file), "--write"])
        assert res2.exit_code == 0
        assert call_count == 1  # Still 1! Zero extra calls!
        assert "MIGRATED (Cached)" in res2.stdout
        # File is now updated from cache
        assert target_file.read_text(encoding="utf-8") == modern_code


def test_cli_migrate_cache_invalidates_on_change(tmp_path: Path) -> None:
    """Verifies that changing file content invalidates the cache and triggers a fresh migration."""
    from copilot_cli.cache import MigrationCache

    code_v1 = "from pydantic import BaseModel\nclass A(BaseModel):\n    class Config:\n        orm_mode = True\n"
    code_v2 = "from pydantic import BaseModel\nclass B(BaseModel):\n    class Config:\n        orm_mode = True\n"
    modern = "from pydantic import BaseModel, ConfigDict\nclass M(BaseModel):\n    model_config = ConfigDict(from_attributes=True)\n"

    target_file = tmp_path / "model.py"
    target_file.write_text(code_v1, encoding="utf-8")
    test_cache_file = tmp_path / "test_cache2.json"

    call_count = 0

    async def counting_stream(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        yield {"stage": "start"}
        yield {"stage": "done", "passed": True, "modern_code": modern, "result": {"total_iterations": 1}}

    with patch("copilot_cli.main.MigrationCache", return_value=MigrationCache(cache_file=test_cache_file)), \
         patch("copilot_cli.main.MigrationOrchestrator.migrate_stream", side_effect=counting_stream):

        # Populate cache for code_v1
        runner.invoke(app, ["migrate", str(target_file), "--diff"])
        assert call_count == 1

        # Modify file content
        target_file.write_text(code_v2, encoding="utf-8")

        # Must invalidate cache and call orchestrator again
        runner.invoke(app, ["migrate", str(target_file), "--diff"])
        assert call_count == 2


def test_cli_migrate_default_current_directory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies copilot migrate defaults to current directory (Path('.')) when path argument is omitted."""
    monkeypatch.chdir(tmp_path)
    clean_file = tmp_path / "app.py"
    clean_file.write_text("x: int = 1\n", encoding="utf-8")

    result = runner.invoke(app, ["migrate", "--diff"])
    assert result.exit_code == 0
    assert "SKIPPED (Clean)" in result.stdout


def test_cli_migrate_run_tests_forces_write(tmp_path: Path) -> None:
    """Verifies passing --run-tests without --diff automatically forces write=True."""
    legacy_code = (
        "from pydantic import BaseModel\n\n"
        "class Model(BaseModel):\n"
        "    class Config:\n"
        "        orm_mode = True\n"
    )
    modern_code = (
        "from pydantic import BaseModel, ConfigDict\n\n"
        "class Model(BaseModel):\n"
        "    model_config = ConfigDict(from_attributes=True)\n"
    )

    target_file = tmp_path / "model.py"
    target_file.write_text(legacy_code, encoding="utf-8")

    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    test_file = tests_dir / "test_model.py"
    test_file.write_text("def test_ok(): assert True\n", encoding="utf-8")

    async def fake_stream(*args, **kwargs):
        yield {"stage": "start"}
        yield {"stage": "done", "passed": True, "modern_code": modern_code, "result": {"total_iterations": 1}}

    with patch("copilot_cli.main.MigrationOrchestrator.migrate_stream", side_effect=fake_stream):
        # Do not pass --write; --run-tests should automatically ensure write=True
        result = runner.invoke(app, ["migrate", str(tmp_path), "--run-tests"])
        assert result.exit_code == 0
        assert "MIGRATED" in result.stdout
        assert target_file.read_text(encoding="utf-8") == modern_code
