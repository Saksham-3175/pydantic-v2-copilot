"""Typer CLI entrypoint for Pydantic v2 Migration Copilot."""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)

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


def _find_test_path(resolved_path: Path) -> Path:
    """Locate tests directory: check resolved_path/tests, resolved_path.parent/tests, and search upwards for git repo root containing tests."""
    if resolved_path.is_dir() and resolved_path.name == "tests":
        return resolved_path
    if resolved_path.is_dir() and (resolved_path / "tests").is_dir():
        return resolved_path / "tests"
    if (resolved_path.parent / "tests").is_dir():
        return resolved_path.parent / "tests"
    curr = resolved_path if resolved_path.is_dir() else resolved_path.parent
    for parent in [curr, *curr.parents]:
        if (parent / ".git").exists():
            if (parent / "tests").is_dir():
                return parent / "tests"
            break
        if (parent / "tests").is_dir():
            return parent / "tests"
    return resolved_path if resolved_path.is_dir() else resolved_path.parent


def _check_or_prompt_api_keys() -> None:
    """Check if GROQ_API_KEY and GEMINI_API_KEY are configured, prompting interactively if missing."""
    if os.environ.get("GROQ_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return

    if sys.stdin.isatty():
        panel_content = (
            "[bold yellow]No LLM API keys detected (GROQ_API_KEY or GEMINI_API_KEY).[/bold yellow]\n\n"
            "• [bold cyan]Offline Deterministic AST Mode[/bold cyan]: Fast, hermetic, rule-based modernizer (zero external API calls).\n"
            "• [bold green]LLM Augmented Mode[/bold green]: Multi-agent synthesis & reflection via Groq (qwen/qwen3.8-27b) or Gemini (gemini-2.5-flash)."
        )
        console.print(Panel(panel_content, title="Copilot Execution Mode", border_style="yellow"))
        key = typer.prompt(
            "Enter GROQ_API_KEY now to enable LLM synthesis (or press Enter to proceed in Offline Mode)",
            default="",
            show_default=False,
            hide_input=True,
        )
        clean_key = key.strip()
        if clean_key:
            os.environ["GROQ_API_KEY"] = clean_key
            env_file = Path(".env")
            try:
                if env_file.exists():
                    env_text = env_file.read_text(encoding="utf-8")
                    if "GROQ_API_KEY=" in env_text:
                        lines = [
                            f"GROQ_API_KEY={clean_key}" if line.startswith("GROQ_API_KEY=") else line
                            for line in env_text.splitlines()
                        ]
                        env_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
                    else:
                        env_file.write_text(env_text.rstrip() + f"\nGROQ_API_KEY={clean_key}\n", encoding="utf-8")
                else:
                    env_file.write_text(f"GROQ_API_KEY={clean_key}\n", encoding="utf-8")
                console.print("[green]✔ GROQ_API_KEY configured and saved to .env[/green]\n")
            except Exception:
                console.print("[green]✔ GROQ_API_KEY configured for current session.[/green]\n")
        else:
            console.print("[cyan]Operating in Offline Deterministic AST Mode (zero external API calls)[/cyan]\n")
    else:
        console.print("[dim]Operating in Offline Deterministic AST Mode (zero external API calls)[/dim]")


def version_callback(value: bool) -> None:
    if value:
        version()
        raise typer.Exit(code=0)


@app.callback()
def root(
    version_flag: Optional[bool] = typer.Option(
        None,
        "--version",
        "-v",
        help="Display version, engine architecture, and active LLM configuration.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Autonomous Multi-Agent AST Modernizer & Critic for Pydantic v2."""
    pass


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
        Path("."),
        help="Path to Python file or directory to modernize (default: current directory)",
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
    _check_or_prompt_api_keys()

    # If run_tests is requested and diff/dry-run is not explicitly enabled, ensure write=True
    if run_tests and not (diff or dry_run):
        write = True

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
        target_test_path = _find_test_path(resolved_path)
        test_root = resolved_path if resolved_path.is_dir() else resolved_path.parent

        console.print(f"\n[bold cyan]Running verification test suite on {target_test_path}...[/bold cyan]\n")
        cmd = [sys.executable, "-m", "pytest", "-o", "addopts=", str(target_test_path)]
        env = os.environ.copy()
        # Add target_dir to PYTHONPATH so test imports resolve properly
        target_root_str = str(test_root)
        env["PYTHONPATH"] = f"{target_root_str}:{env.get('PYTHONPATH', '')}"

        proc = subprocess.run(cmd, env=env, check=False)
        if proc.returncode != 0:
            console.print(f"\n[bold red]✖ Target test suite failed with exit code {proc.returncode}[/bold red]")
            console.print(
                "[yellow]Note: If failure is due to missing dependencies in the target project, "
                "ensure the target virtual environment or dependencies are installed and available.[/yellow]"
            )
            raise typer.Exit(code=proc.returncode)
        else:
            console.print("\n[bold green]✔ Target test suite passed with 100% success![/bold green]")
