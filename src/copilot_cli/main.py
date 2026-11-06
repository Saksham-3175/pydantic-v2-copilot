"""Typer CLI entrypoint for Pydantic v2 Migration Copilot."""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.panel import Panel
from rich.table import Table

from copilot_cli.cache import MigrationCache
from copilot_cli.screener import FileScanResult, scan_directory, scan_file
from copilot_cli.ui import (
    FileMigrationSummary,
    console,
    print_banner,
    render_diff,
    render_scan_results,
    render_scorecard,
)
from core_agents.orchestrator import MigrationOrchestrator

app = typer.Typer(
    name="copilot",
    help="Autonomous Multi-Agent AST Modernizer & Critic for Pydantic v2",
    no_args_is_help=True,
)


def _check_git_uncommitted(target_path: Path) -> bool:
    """Check if the repository containing target_path has uncommitted changes."""
    try:
        search_dir = target_path if target_path.is_dir() else target_path.parent
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(search_dir),
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode == 0 and res.stdout.strip():
            return True
    except Exception:
        pass
    return False


@app.command()
def version() -> None:
    """Display version, engine architecture, and active LLM configuration."""
    print_banner()
    groq_key = os.environ.get("GROQ_API_KEY")
    gemini_key = os.environ.get("GEMINI_API_KEY")
    groq_status = "[green]Configured[/green]" if groq_key else "[dim]Not set[/dim]"
    gemini_status = "[green]Configured[/green]" if gemini_key else "[dim]Not set[/dim]"

    info_table = Table(title="Copilot System Architecture", border_style="cyan")
    info_table.add_column("Component", style="bold cyan")
    info_table.add_column("Specification", style="white")

    info_table.add_row("CLI Version", "0.1.0")
    info_table.add_row("Multi-Agent Architecture", "Decomposer ➔ Scoped Qdrant RAG ➔ Synthesizer ➔ Critic")
    info_table.add_row("Deterministic Critic", "4-Gate AST & Runtime Invariant Engine")
    info_table.add_row("Primary Synthesizer (Groq)", f"qwen/qwen3.8-27b ({groq_status})")
    info_table.add_row("Fallback Synthesizer (Gemini)", f"gemini-2.5-flash ({gemini_status})")
    info_table.add_row("Vector Knowledge Engine", "Embedded Qdrant (python_tooling_docs)")

    console.print(info_table)


@app.command()
def scan(
    path: Path = typer.Argument(
        Path("."),
        help="Path to Python file or directory to scan",
        exists=True,
    ),
) -> None:
    """Scan Python files and detect legacy Pydantic v1 patterns."""
    print_banner()
    resolved_path = path.resolve()

    if resolved_path.is_file():
        scan_results = [scan_file(resolved_path)]
    else:
        with console.status(f"[bold cyan]Scanning {resolved_path.name} with 2-Stage AST Screener...[/bold cyan]"):
            scan_results = scan_directory(resolved_path)

    base_dir = resolved_path if resolved_path.is_dir() else resolved_path.parent
    render_scan_results(scan_results, base_path=base_dir)


async def _run_migration_stream(
    file_path: Path,
    orchestrator: MigrationOrchestrator,
    max_iterations: int,
) -> tuple[bool, str, int, Optional[str]]:
    """Stream migration for a single file through the orchestrator reflection loop."""
    code = file_path.read_text(encoding="utf-8")
    final_modern_code = code
    passed = False
    total_iterations = 1
    error_msg: Optional[str] = None

    with console.status(f"[bold cyan]Modernizing {file_path.name}...[/bold cyan]") as status:
        async for event in orchestrator.migrate_stream(code, max_iterations=max_iterations):
            stage = event.get("stage")
            if stage == "start":
                status.update(f"[bold blue][1/4][/bold blue] Initializing analysis: {file_path.name}")
            elif stage == "decomposed":
                topics = event.get("detected_topics", [])
                topics_str = ", ".join(topics[:2]) if topics else "general"
                status.update(f"[bold cyan][2/4][/bold cyan] Decomposed topics: {topics_str}")
            elif stage == "retrieved":
                count = event.get("count", 0)
                status.update(f"[bold yellow][3/4][/bold yellow] Retrieved {count} doc chunks from Qdrant")
            elif stage == "synthesizing":
                it = event.get("iteration", 1)
                status.update(f"[bold magenta][4/4][/bold magenta] Synthesizing modern code (attempt {it})...")
            elif stage == "critic_verification":
                it = event.get("iteration", 1)
                status.update(f"[bold cyan][Critic][/bold cyan] Verifying AST invariants (attempt {it})...")
            elif stage == "done":
                passed = event.get("passed", False)
                final_modern_code = event.get("modern_code", code)
                res_data = event.get("result", {})
                total_iterations = res_data.get("total_iterations", 1)
                if not passed:
                    verdict = res_data.get("verdict", {})
                    failures = verdict.get("failures", [])
                    if failures:
                        error_msg = failures[0].get("message", "Critic verification rejected draft")
                    else:
                        error_msg = "Critic verification rejected draft"

    return passed, final_modern_code, total_iterations, error_msg


@app.command()
def migrate(
    path: Path = typer.Argument(
        ...,
        help="Path to Python file or directory to modernize",
        exists=True,
    ),
    diff: bool = typer.Option(
        False,
        "--diff",
        help="Preview unified diffs without modifying files",
    ),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Alias for --diff (preview mode)",
    ),
    write: bool = typer.Option(
        False,
        "--write",
        "-w",
        help="Write modern code directly to files on disk",
    ),
    run_tests: bool = typer.Option(
        False,
        "--run-tests",
        "-t",
        help="Run pytest on target repository after migration",
    ),
    max_iterations: int = typer.Option(
        2,
        "--max-iterations",
        "-i",
        help="Max reflection attempts per file (default: 2)",
    ),
) -> None:
    """Modernize legacy Pydantic v1 code to Pydantic v2."""
    print_banner()
    resolved_path = path.resolve()

    # Default to diff/preview if write is not requested
    is_preview = not write or diff or dry_run

    if write:
        if _check_git_uncommitted(resolved_path):
            console.print(
                "[yellow]warning:[/yellow] uncommitted changes in git repository (commit or stash before applying --write)"
            )

    # 1. Screen target for legacy patterns
    if resolved_path.is_file():
        scan_results = [scan_file(resolved_path)]
    else:
        with console.status("[bold cyan]Running 2-Stage AST Screener...[/bold cyan]"):
            scan_results = scan_directory(resolved_path)

    summaries: list[FileMigrationSummary] = []
    orchestrator = MigrationOrchestrator(max_iterations=max_iterations)
    cache = MigrationCache()

    for scan_res in scan_results:
        f_path = scan_res.file_path
        if not scan_res.has_legacy_patterns:
            # File is completely clean - skip zero LLM tokens
            summaries.append(
                FileMigrationSummary(
                    file_path=str(f_path),
                    status="SKIPPED (Clean)",
                    iterations=1,
                    passed=True,
                )
            )
            continue

        original_code = f_path.read_text(encoding="utf-8")
        cached = cache.get(f_path, original_code)

        if cached is not None:
            modern_code = cached["modern_code"]
            iterations = cached.get("iterations", 1)
            passed = True
            err = None
            if is_preview:
                render_diff(f_path, original_code, modern_code)
                action_status = "DRY-RUN (Cached)"
            else:
                f_path.write_text(modern_code, encoding="utf-8")
                console.print(f"[bold green]✔ Modernized & written (cached):[/bold green] {f_path}")
                action_status = "MIGRATED (Cached)"
        else:
            passed, modern_code, iterations, err = asyncio.run(
                _run_migration_stream(
                    file_path=f_path,
                    orchestrator=orchestrator,
                    max_iterations=max_iterations,
                )
            )

            if is_preview:
                render_diff(f_path, original_code, modern_code)
                action_status = "DRY-RUN" if passed else "FAILED"
            else:
                if passed:
                    f_path.write_text(modern_code, encoding="utf-8")
                    console.print(f"[bold green]✔ Modernized & written:[/bold green] {f_path}")
                    action_status = "MIGRATED"
                else:
                    console.print(f"[bold red]✖ Verification failed, file unchanged:[/bold red] {f_path}")
                    action_status = "FAILED"

            if passed:
                cache.set(f_path, original_code, modern_code, iterations)

        summaries.append(
            FileMigrationSummary(
                file_path=str(f_path),
                status=action_status,
                iterations=iterations,
                passed=passed,
                error=err,
            )
        )

    # 2. Render Scorecard Table
    base_dir = resolved_path if resolved_path.is_dir() else resolved_path.parent
    render_scorecard(summaries, base_path=base_dir)

    # 3. Optional pytest execution
    if run_tests:
        test_dir = resolved_path if resolved_path.is_dir() else resolved_path.parent
        candidate_tests = test_dir / "tests"
        target_test_path = candidate_tests if candidate_tests.exists() else test_dir

        console.print(f"\n[bold cyan]Running verification test suite on {target_test_path}...[/bold cyan]\n")
        cmd = [sys.executable, "-m", "pytest", str(target_test_path)]
        env = os.environ.copy()
        # Add target_dir to PYTHONPATH so test imports resolve properly
        target_root = str(test_dir)
        env["PYTHONPATH"] = f"{target_root}:{env.get('PYTHONPATH', '')}"

        proc = subprocess.run(cmd, env=env, check=False)
        if proc.returncode != 0:
            console.print(f"\n[bold red]✖ Target test suite failed with exit code {proc.returncode}[/bold red]")
            raise typer.Exit(code=proc.returncode)
        else:
            console.print("\n[bold green]✔ Target test suite passed with 100% success![/bold green]")
