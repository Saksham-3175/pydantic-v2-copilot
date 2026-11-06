"""Fast 2-Stage Deterministic AST Screener for legacy Pydantic v1 and FastAPI patterns."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Optional, Sequence


# Stage 1: Quick substring / token filter on raw source code for fast rejection
STAGE_1_MARKERS: tuple[str, ...] = (
    "Config",
    "validator",
    "root_validator",
    "BaseSettings",
    ".dict(",
    ".json(",
    "on_event",
    "regex=",
)

# Canonical pattern types
PATTERN_CLASS_CONFIG = "class_config"
PATTERN_VALIDATOR = "validator"
PATTERN_ROOT_VALIDATOR = "root_validator"
PATTERN_LEGACY_EXPORT = "legacy_export"
PATTERN_BASE_SETTINGS = "base_settings"
PATTERN_ON_EVENT = "on_event"
PATTERN_FIELD_REGEX = "field_regex"

# Directories ignored by default during folder scanning
DEFAULT_IGNORED_DIRS: set[str] = {
    ".venv",
    "venv",
    "__pycache__",
    ".git",
    ".pytest_cache",
    ".agents",
    "dist",
    "build",
}


@dataclass
class PatternMatch:
    """Represents a single legacy pattern match found in source code."""

    line: int
    pattern_type: str
    description: str
    snippet: Optional[str] = None


@dataclass
class FileScanResult:
    """Result of scanning a Python file or code snippet for legacy patterns."""

    file_path: Path
    has_legacy_patterns: bool
    matches: list[PatternMatch] = field(default_factory=list)
    error: Optional[str] = None


class _LegacyPatternVisitor(ast.NodeVisitor):
    """Deterministic AST Visitor detecting canonical Pydantic v1 / legacy FastAPI patterns."""

    def __init__(self, code_lines: list[str]) -> None:
        self.code_lines = code_lines
        self.matches: list[PatternMatch] = []
        self._class_depth = 0

    def _get_snippet(self, lineno: int) -> Optional[str]:
        if 1 <= lineno <= len(self.code_lines):
            return self.code_lines[lineno - 1].strip()
        return None

    @staticmethod
    def _get_decorator_name(decorator: ast.expr) -> str:
        """Extract the identifier or attribute name from a decorator expression."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr
        return ""

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Check for inner class Config: in any class
        if self._class_depth > 0 and node.name == "Config":
            self.matches.append(
                PatternMatch(
                    line=node.lineno,
                    pattern_type=PATTERN_CLASS_CONFIG,
                    description="Legacy inner 'class Config:' is deprecated in Pydantic v2. Use 'model_config = ConfigDict(...)'.",
                    snippet=self._get_snippet(node.lineno),
                )
            )

        # Check for inheritance directly from pydantic.BaseSettings
        for base in node.bases:
            if isinstance(base, ast.Attribute) and base.attr == "BaseSettings":
                if isinstance(base.value, ast.Name) and base.value.id == "pydantic":
                    self.matches.append(
                        PatternMatch(
                            line=base.lineno if hasattr(base, "lineno") else node.lineno,
                            pattern_type=PATTERN_BASE_SETTINGS,
                            description="'pydantic.BaseSettings' is deprecated. Inherit from 'pydantic_settings.BaseSettings'.",
                            snippet=self._get_snippet(base.lineno if hasattr(base, "lineno") else node.lineno),
                        )
                    )

        self._class_depth += 1
        self.generic_visit(node)
        self._class_depth -= 1

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._check_function_decorators(node)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._check_function_decorators(node)
        self.generic_visit(node)

    def _check_function_decorators(
        self, node: ast.FunctionDef | ast.AsyncFunctionDef
    ) -> None:
        for dec in node.decorator_list:
            dec_name = self._get_decorator_name(dec)
            dec_line = dec.lineno if hasattr(dec, "lineno") else node.lineno

            # 1. @validator decorator
            if dec_name == "validator":
                self.matches.append(
                    PatternMatch(
                        line=dec_line,
                        pattern_type=PATTERN_VALIDATOR,
                        description="Legacy '@validator' decorator is deprecated. Use '@field_validator' with '@classmethod'.",
                        snippet=self._get_snippet(dec_line),
                    )
                )
            # 2. @root_validator decorator
            elif dec_name == "root_validator":
                self.matches.append(
                    PatternMatch(
                        line=dec_line,
                        pattern_type=PATTERN_ROOT_VALIDATOR,
                        description="Legacy '@root_validator' decorator is deprecated. Use '@model_validator'.",
                        snippet=self._get_snippet(dec_line),
                    )
                )

            # 3. @app.on_event("startup" / "shutdown") or .on_event decorator calls
            is_on_event = False
            if isinstance(dec, ast.Call):
                if isinstance(dec.func, ast.Attribute) and dec.func.attr == "on_event":
                    is_on_event = True
                elif isinstance(dec.func, ast.Name) and dec.func.id == "on_event":
                    is_on_event = True
            elif isinstance(dec, ast.Attribute) and dec.attr == "on_event":
                is_on_event = True

            if is_on_event:
                event_arg = ""
                if (
                    isinstance(dec, ast.Call)
                    and dec.args
                    and isinstance(dec.args[0], ast.Constant)
                    and isinstance(dec.args[0].value, str)
                ):
                    event_arg = f" ('{dec.args[0].value}')"
                self.matches.append(
                    PatternMatch(
                        line=dec_line,
                        pattern_type=PATTERN_ON_EVENT,
                        description=f"Legacy FastAPI '.on_event{event_arg}' decorator is deprecated. Use lifespan async context manager instead.",
                        snippet=self._get_snippet(dec_line),
                    )
                )

    def visit_Call(self, node: ast.Call) -> None:
        # Legacy export method calls: .dict() or .json() on variables/objects
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("dict", "json"):
            attr_name = node.func.attr
            target_method = ".model_dump()" if attr_name == "dict" else ".model_dump_json()"
            self.matches.append(
                PatternMatch(
                    line=node.lineno,
                    pattern_type=PATTERN_LEGACY_EXPORT,
                    description=f"Legacy export method '.{attr_name}()' is deprecated in Pydantic v2. Use '{target_method}' instead.",
                    snippet=self._get_snippet(node.lineno),
                )
            )

        # Field regex: Field(..., regex=...)
        is_field_call = False
        if isinstance(node.func, ast.Name) and node.func.id == "Field":
            is_field_call = True
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "Field":
            is_field_call = True

        if is_field_call:
            for kw in node.keywords:
                if kw.arg == "regex":
                    self.matches.append(
                        PatternMatch(
                            line=node.lineno,
                            pattern_type=PATTERN_FIELD_REGEX,
                            description="Field(regex=...) is deprecated in Pydantic v2. Use Field(pattern=...) instead.",
                            snippet=self._get_snippet(node.lineno),
                        )
                    )

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # from pydantic import BaseSettings (or from pydantic.something import BaseSettings)
        if node.module and (node.module == "pydantic" or node.module.startswith("pydantic.")):
            for alias in node.names:
                if alias.name == "BaseSettings":
                    self.matches.append(
                        PatternMatch(
                            line=node.lineno,
                            pattern_type=PATTERN_BASE_SETTINGS,
                            description="'BaseSettings' has been moved to 'pydantic_settings'. Import from 'pydantic_settings import BaseSettings'.",
                            snippet=self._get_snippet(node.lineno),
                        )
                    )
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            if alias.name == "pydantic.BaseSettings":
                self.matches.append(
                    PatternMatch(
                        line=node.lineno,
                        pattern_type=PATTERN_BASE_SETTINGS,
                        description="'BaseSettings' has been moved to 'pydantic_settings'. Import from 'pydantic_settings import BaseSettings'.",
                        snippet=self._get_snippet(node.lineno),
                    )
                )
        self.generic_visit(node)


def scan_code(code: str, file_path: Optional[Path] = None) -> FileScanResult:
    """Scan code string using a 2-stage screener engine.
    
    Stage 1: Quick substring / token filter on raw source code for fast rejection.
    Stage 2: Deterministic AST scan checking for canonical breaking changes.
    """
    target_path = file_path if file_path is not None else Path("<string>")

    # Stage 1: Quick substring / token filter for fast rejection
    if not any(marker in code for marker in STAGE_1_MARKERS):
        return FileScanResult(
            file_path=target_path,
            has_legacy_patterns=False,
            matches=[],
            error=None,
        )

    # Stage 2: Deterministic AST scan
    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return FileScanResult(
            file_path=target_path,
            has_legacy_patterns=False,
            matches=[],
            error=f"SyntaxError on line {exc.lineno}: {exc.msg}",
        )

    code_lines = code.splitlines()
    visitor = _LegacyPatternVisitor(code_lines=code_lines)
    visitor.visit(tree)

    visitor.matches.sort(key=lambda m: (m.line, m.pattern_type))
    has_legacy = len(visitor.matches) > 0

    return FileScanResult(
        file_path=target_path,
        has_legacy_patterns=has_legacy,
        matches=visitor.matches,
        error=None,
    )


def scan_file(file_path: Path) -> FileScanResult:
    """Read and scan a single Python file for legacy patterns."""
    path = Path(file_path)
    if not path.is_file():
        return FileScanResult(
            file_path=path,
            has_legacy_patterns=False,
            matches=[],
            error=f"File not found: {path}",
        )
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        return FileScanResult(
            file_path=path,
            has_legacy_patterns=False,
            matches=[],
            error=f"Failed to read file: {exc}",
        )
    return scan_code(content, file_path=path)


def scan_directory(directory_path: Path, recursive: bool = True) -> list[FileScanResult]:
    """Scan Python files in a directory, ignoring virtualenvs, caches, and build folders.
    
    Ignored directories: .venv, venv, __pycache__, .git, .pytest_cache, .agents, dist, build.
    """
    path = Path(directory_path)
    if not path.exists():
        return []
    if path.is_file():
        return [scan_file(path)]

    results: list[FileScanResult] = []

    if recursive:
        for root, dirs, files in os.walk(path):
            dirs[:] = [d for d in dirs if d not in DEFAULT_IGNORED_DIRS]
            for file in sorted(files):
                if file.endswith(".py"):
                    results.append(scan_file(Path(root) / file))
    else:
        for entry in sorted(path.iterdir()):
            if entry.is_file() and entry.name.endswith(".py"):
                results.append(scan_file(entry))

    results.sort(key=lambda r: str(r.file_path))
    return results


__all__ = [
    "PatternMatch",
    "FileScanResult",
    "scan_code",
    "scan_file",
    "scan_directory",
    "PATTERN_CLASS_CONFIG",
    "PATTERN_VALIDATOR",
    "PATTERN_ROOT_VALIDATOR",
    "PATTERN_LEGACY_EXPORT",
    "PATTERN_BASE_SETTINGS",
    "PATTERN_ON_EVENT",
    "PATTERN_FIELD_REGEX",
    "STAGE_1_MARKERS",
    "DEFAULT_IGNORED_DIRS",
]
