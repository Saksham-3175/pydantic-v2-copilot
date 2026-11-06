"""Cinematic terminal UX and visual renderer using Rich."""

from __future__ import annotations

import difflib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

console = Console()


@dataclass
class FileMigrationSummary:
    """Summary of migration operation on a single file."""

    file_path: str
    status: str  # "MIGRATED", "DRY-RUN", "SKIPPED (Clean)", "FAILED"
    iterations: int = 1
    passed: bool = True
    changes_count: int = 0
    error: Optional[str] = None


def print_banner() -> None:
    """Print standard CLI banner."""
    title = Text(" Pydantic v2 Migration Copilot ", style="bold white on blue")
    subtitle = Text("Autonomous Multi-Agent AST Modernizer & Critic", style="dim")
    console.print()
    console.print(Panel(Text.assemble(title, "\n", subtitle), border_style="blue"))
    console.print()


def render_diff(file_path: str | Path, original_code: str, modern_code: str) -> None:
    """Render unified Git diff with syntax highlighting."""
    orig_lines = original_code.splitlines(keepends=True)
    mod_lines = modern_code.splitlines(keepends=True)
    
    diff = list(
        difflib.unified_diff(
            orig_lines,
            mod_lines,
            fromfile=f"a/{Path(file_path).name}",
            tofile=f"b/{Path(file_path).name}",
            n=3,
        )
    )

    if not diff:
        console.print(f"[dim]No changes required for {file_path}[/dim]")
        return

    diff_text = "".join(diff)
    syntax = Syntax(diff_text, "diff", theme="ansi_dark", line_numbers=True)
    panel = Panel(
        syntax,
        title=f"[bold cyan]Diff: {Path(file_path).name}[/bold cyan]",
        subtitle=f"[dim]{file_path}[/dim]",
        border_style="cyan",
    )
    console.print(panel)


def render_scan_results(scan_results: Sequence[Any], base_path: Optional[Path] = None) -> None:
    """Render repository AST scan results in a rich table."""
    table = Table(
        title="Repository AST Scan Results",
        header_style="bold magenta",
        show_header=True,
        border_style="dim",
    )
    table.add_column("File", style="cyan")
    table.add_column("Status", justify="center")
    table.add_column("Detected Legacy Invariants", style="yellow")

    flagged_count = 0
    clean_count = 0

    for res in scan_results:
        # FileScanResult duck-typing
        p = Path(getattr(res, "file_path", res))
        if base_path:
            try:
                file_path_str = str(p.relative_to(base_path))
            except Exception:
                file_path_str = p.name
        else:
            file_path_str = p.name
        has_patterns = getattr(res, "has_legacy_patterns", False)
        matches = getattr(res, "matches", [])

        if has_patterns:
            flagged_count += 1
            status_text = "[bold red]NEEDS MIGRATION[/bold red]"
            details = ", ".join(
                f"{getattr(m, 'pattern_type', 'pattern')} (L{getattr(m, 'line', '?')})"
                for m in matches
            ) or "Legacy patterns detected"
        else:
            clean_count += 1
            status_text = "[bold green]CLEAN[/bold green]"
            details = "[dim]0 legacy patterns (Skipped)[/dim]"

        table.add_row(file_path_str, status_text, details)

    console.print(table)
    console.print()

    summary_msg = (
        f"Scanned [bold]{len(scan_results)}[/bold] files: "
        f"[bold red]{flagged_count}[/bold red] need migration, "
        f"[bold green]{clean_count}[/bold green] clean."
    )
    console.print(Panel(summary_msg, border_style="blue" if flagged_count == 0 else "yellow"))


def render_scorecard(summaries: Sequence[FileMigrationSummary], base_path: Optional[Path] = None) -> None:
    """Render migration execution scorecard table."""
    table = Table(
        title="Migration Execution Scorecard",
        header_style="bold green",
        show_header=True,
        border_style="green",
    )
    table.add_column("Target File", style="cyan")
    table.add_column("Action / Mode", justify="center")
    table.add_column("Iterations", justify="center")
    table.add_column("Critic Verdict", justify="center")

    total = len(summaries)
    migrated = 0
    skipped = 0
    failed = 0

    for s in summaries:
        if s.status.startswith("MIGRATED") or s.status.startswith("DRY-RUN"):
            migrated += 1
            mode_style = "bold green" if "MIGRATED" in s.status else "bold yellow"
            mode_display = f"[{mode_style}]{s.status}[/{mode_style}]"
            iter_display = str(s.iterations)
            verdict_display = (
                "[bold green]PASSED (4/4 Gates)[/bold green]"
                if s.passed
                else f"[bold red]FAILED: {s.error or 'Critic rejection'}[/bold red]"
            )
        elif s.status == "SKIPPED (Clean)":
            skipped += 1
            mode_display = "[dim yellow]SKIPPED (Clean)[/dim yellow]"
            iter_display = "[dim]—[/dim]"
            verdict_display = "[dim green]CLEAN (Bypassed)[/dim green]"
        else:
            failed += 1
            mode_display = "[bold red]FAILED[/bold red]"
            iter_display = str(s.iterations)
            verdict_display = f"[bold red]FAILED: {s.error or 'Critic rejection'}[/bold red]"

        p = Path(s.file_path)
        if base_path:
            try:
                file_display = str(p.relative_to(base_path))
            except Exception:
                file_display = p.name
        else:
            file_display = p.name

        table.add_row(
            file_display,
            mode_display,
            iter_display,
            verdict_display,
        )

    console.print()
    console.print(table)
    console.print()

    pass_rate = 100.0 if (migrated + skipped + failed) == 0 else ((migrated + skipped) / total) * 100.0
    summary_text = (
        f"Total Files Scanned: [bold]{total}[/bold] | "
        f"Migrated: [bold green]{migrated}[/bold green] | "
        f"Skipped (Clean): [bold yellow]{skipped}[/bold yellow] | "
        f"Failed: [bold red]{failed}[/bold red] | "
        f"Success Rate: [bold cyan]{pass_rate:.1f}%[/bold cyan]"
    )
    console.print(Panel(summary_text, title="Scorecard Summary", border_style="cyan"))
