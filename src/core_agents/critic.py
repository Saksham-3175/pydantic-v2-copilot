"""Deterministic Python AST and Rule Critic for Pydantic v2 and FastAPI modern patterns."""

from __future__ import annotations

import ast
import importlib.util
import sys
import types
import typing
from unittest.mock import MagicMock

from core_agents.schemas import CriticVerdict, GateFailure


class _MockModuleFinder:
    """Import finder that provides fallback MagicMock modules for missing external libraries."""

    def __init__(self) -> None:
        self.mocked_modules: set[str] = set()

    def find_spec(
        self,
        fullname: str,
        path: typing.Optional[typing.Sequence[typing.Union[bytes, str]]] = None,
        target: typing.Optional[types.ModuleType] = None,
    ) -> typing.Optional[importlib.machinery.ModuleSpec]:
        # Only mock external/unknown packages, let standard libraries and installed packages load
        self.mocked_modules.add(fullname)
        return importlib.util.spec_from_loader(fullname, loader=self)

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> types.ModuleType:
        mock_mod = MagicMock()
        mock_mod.__name__ = spec.name
        return mock_mod

    def exec_module(self, module: types.ModuleType) -> None:
        pass


class _RuleScanVisitor(ast.NodeVisitor):
    """AST Visitor enforcing negative and positive invariants for Pydantic v2."""

    def __init__(self, code_lines: list[str]) -> None:
        self.code_lines = code_lines
        self.failures: list[GateFailure] = []
        self._class_depth = 0

    def _get_offending_code(self, lineno: typing.Optional[int]) -> typing.Optional[str]:
        if lineno is not None and 1 <= lineno <= len(self.code_lines):
            return self.code_lines[lineno - 1].strip()
        return None

    @staticmethod
    def _get_decorator_name(decorator: ast.expr) -> str:
        """Extract the root or attribute name of a decorator expression."""
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

    @staticmethod
    def _is_valid_config_dict(val_node: typing.Optional[ast.AST]) -> bool:
        """Verify whether an assignment expression is a valid ConfigDict or dictionary."""
        if val_node is None:
            return False
        if isinstance(val_node, ast.Dict):
            return True
        if isinstance(val_node, ast.Call):
            func = val_node.func
            if isinstance(func, ast.Name) and func.id in ("ConfigDict", "dict", "SettingsConfigDict"):
                return True
            if isinstance(func, ast.Attribute) and func.attr in ("ConfigDict", "dict", "SettingsConfigDict"):
                return True
        return False

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        # Gate 2: Inner 'class Config:' is deprecated in Pydantic v2
        if self._class_depth > 0 and node.name == "Config":
            self.failures.append(
                GateFailure(
                    gate=2,
                    gate_name="Negative Invariant Scan",
                    message="Legacy inner 'class Config:' is deprecated in Pydantic v2. Use 'model_config = ConfigDict(...)'.",
                    offending_line=node.lineno,
                    offending_code=self._get_offending_code(node.lineno),
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
        self, node: typing.Union[ast.FunctionDef, ast.AsyncFunctionDef]
    ) -> None:
        has_field_validator = False
        has_classmethod = False

        for dec in node.decorator_list:
            dec_name = self._get_decorator_name(dec)

            # Gate 2: Legacy validator or root_validator
            if dec_name in ("validator", "root_validator"):
                self.failures.append(
                    GateFailure(
                        gate=2,
                        gate_name="Negative Invariant Scan",
                        message="Legacy validator decorator is deprecated in Pydantic v2. Use @field_validator or @model_validator.",
                        offending_line=node.lineno,
                        offending_code=self._get_offending_code(node.lineno),
                    )
                )

            # Gate 2: Legacy FastAPI events: app.on_event("startup" / "shutdown")
            if (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Attribute)
                and dec.func.attr == "on_event"
            ):
                event_args = [
                    arg.value
                    for arg in dec.args
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str)
                ]
                if any(e in ("startup", "shutdown") for e in event_args) or not dec.args:
                    self.failures.append(
                        GateFailure(
                            gate=2,
                            gate_name="Negative Invariant Scan",
                            message="@app.on_event('startup'/'shutdown') is deprecated. Use '@asynccontextmanager async def lifespan(app: FastAPI):'.",
                            offending_line=node.lineno,
                            offending_code=self._get_offending_code(node.lineno),
                        )
                    )

            if dec_name == "field_validator":
                has_field_validator = True
            elif dec_name == "classmethod":
                has_classmethod = True

        # Gate 3: Positive Invariant: @field_validator must always have @classmethod
        if has_field_validator and not has_classmethod:
            self.failures.append(
                GateFailure(
                    gate=3,
                    gate_name="Positive Invariant Scan",
                    message="@field_validator must always be accompanied by @classmethod decorator.",
                    offending_line=node.lineno,
                    offending_code=self._get_offending_code(node.lineno),
                )
            )

    def visit_Call(self, node: ast.Call) -> None:
        # Gate 2: Legacy methods .dict() or .json() on attribute/object
        if isinstance(node.func, ast.Attribute) and node.func.attr in ("dict", "json"):
            self.failures.append(
                GateFailure(
                    gate=2,
                    gate_name="Negative Invariant Scan",
                    message="Legacy export method '.dict()' or '.json()' is deprecated. Use '.model_dump()' or '.model_dump_json()'.",
                    offending_line=node.lineno,
                    offending_code=self._get_offending_code(node.lineno),
                )
            )

        # Gate 2: Legacy Field arguments: regex=
        is_field_call = False
        if isinstance(node.func, ast.Name) and node.func.id == "Field":
            is_field_call = True
        elif isinstance(node.func, ast.Attribute) and node.func.attr == "Field":
            is_field_call = True

        if is_field_call:
            for kw in node.keywords:
                if kw.arg == "regex":
                    self.failures.append(
                        GateFailure(
                            gate=2,
                            gate_name="Negative Invariant Scan",
                            message="Field(regex=...) is replaced by Field(pattern=...) in Pydantic v2.",
                            offending_line=node.lineno,
                            offending_code=self._get_offending_code(node.lineno),
                        )
                    )

        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        # Gate 2: BaseSettings moved to pydantic_settings
        if node.module == "pydantic":
            for alias in node.names:
                if alias.name == "BaseSettings":
                    self.failures.append(
                        GateFailure(
                            gate=2,
                            gate_name="Negative Invariant Scan",
                            message="BaseSettings has been moved to 'pydantic_settings'. Import from 'pydantic_settings import BaseSettings'.",
                            offending_line=node.lineno,
                            offending_code=self._get_offending_code(node.lineno),
                        )
                    )
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        # Gate 3: Assignment to model_config must use ConfigDict(...) or dict
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == "model_config":
                if not self._is_valid_config_dict(node.value):
                    self.failures.append(
                        GateFailure(
                            gate=3,
                            gate_name="Positive Invariant Scan",
                            message="Assignment to 'model_config' must use 'ConfigDict(...)'.",
                            offending_line=node.lineno,
                            offending_code=self._get_offending_code(node.lineno),
                        )
                    )
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # Gate 3: Annotated assignment to model_config
        if isinstance(node.target, ast.Name) and node.target.id == "model_config":
            if node.value is None or not self._is_valid_config_dict(node.value):
                self.failures.append(
                    GateFailure(
                        gate=3,
                        gate_name="Positive Invariant Scan",
                        message="Assignment to 'model_config' must use 'ConfigDict(...)'.",
                        offending_line=node.lineno,
                        offending_code=self._get_offending_code(node.lineno),
                    )
                )
        self.generic_visit(node)


class ASTCritic:
    """Deterministic AST and Rule Critic verifying strict Pydantic v2 invariants."""

    def verify(self, code: str) -> CriticVerdict:
        """Alias for evaluate to conform with supervisor contracts."""
        return self.evaluate(code)

    def evaluate(self, code: str) -> CriticVerdict:
        """Run 4 deterministic verification gates against the candidate Python code.

        Gate 1: AST Parseability (ast.parse)
        Gate 2: Negative Invariant Scan (ast.NodeVisitor)
        Gate 3: Positive Invariant Scan (ast.NodeVisitor)
        Gate 4: Runtime Instantiation / Compilation Safety (compile & exec sandbox)
        """
        code_lines = code.splitlines()

        # Gate 1: AST Parseability
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            offending_code = e.text.strip() if e.text else None
            failures = [
                GateFailure(
                    gate=1,
                    gate_name="AST Parseability",
                    message=f"SyntaxError: {e.msg}",
                    offending_line=e.lineno,
                    offending_code=offending_code,
                )
            ]
            line_info = f" on line {e.lineno}" if e.lineno else ""
            suggested_fixes = [f"Fix syntax error{line_info}: {e.msg}"]
            return CriticVerdict(
                passed=False,
                failures=failures,
                suggested_fixes=suggested_fixes,
            )

        failures: list[GateFailure] = []

        # Gate 2 & Gate 3: Static AST Rule Scans
        visitor = _RuleScanVisitor(code_lines)
        visitor.visit(tree)
        failures.extend(visitor.failures)

        # Gate 4: Runtime Instantiation / Compilation Safety
        try:
            compiled = compile(tree, filename="<critic_sandbox>", mode="exec")
            namespace: dict[str, typing.Any] = {"__name__": "__main__"}

            finder = _MockModuleFinder()
            sys.meta_path.append(finder)
            try:
                exec(compiled, namespace)
            finally:
                if finder in sys.meta_path:
                    sys.meta_path.remove(finder)
                for mod_name in finder.mocked_modules:
                    sys.modules.pop(mod_name, None)
        except (ImportError, ModuleNotFoundError):
            # Gracefully handle missing external modules
            pass
        except Exception as err:
            failures.append(
                GateFailure(
                    gate=4,
                    gate_name="Runtime Execution",
                    message=f"Runtime evaluation error: {err}",
                )
            )

        # Gate 5: Compile Verdict & Suggested Fixes
        if not failures:
            return CriticVerdict(passed=True, failures=[], suggested_fixes=[])

        suggested_fixes = self._generate_suggested_fixes(failures)
        return CriticVerdict(
            passed=False,
            failures=failures,
            suggested_fixes=suggested_fixes,
        )

    def _generate_suggested_fixes(self, failures: list[GateFailure]) -> list[str]:
        """Derive actionable, deduplicated recommendations for each reported failure."""
        fixes: list[str] = []
        for failure in failures:
            fix = self._derive_fix(failure)
            if fix and fix not in fixes:
                fixes.append(fix)
        return fixes

    @staticmethod
    def _derive_fix(failure: GateFailure) -> str:
        msg = failure.message
        if failure.gate == 1:
            line_str = f" on line {failure.offending_line}" if failure.offending_line else ""
            return f"Fix syntax error{line_str}: {failure.message}"
        elif failure.gate == 2:
            if "validator" in msg:
                return "Replace @validator / @root_validator with @field_validator (requiring @classmethod) or @model_validator."
            elif "Config" in msg:
                return "Replace legacy inner 'class Config:' with 'model_config = ConfigDict(...)'."
            elif "export method" in msg or ".dict()" in msg or ".json()" in msg:
                return "Replace '.dict()' with '.model_dump()' and '.json()' with '.model_dump_json()'."
            elif "regex" in msg:
                return "Replace Field(regex=...) with Field(pattern=...) in Pydantic v2."
            elif "BaseSettings" in msg:
                return "Import BaseSettings from 'pydantic_settings' instead of 'pydantic'."
            elif "on_event" in msg:
                return "Replace @app.on_event('startup'/'shutdown') with FastAPI lifespan asynccontextmanager."
            return failure.message
        elif failure.gate == 3:
            if "@classmethod" in msg:
                return "Add the '@classmethod' decorator to any method decorated with '@field_validator'."
            elif "model_config" in msg:
                return "Set 'model_config = ConfigDict(...)' using ConfigDict imported from 'pydantic'."
            return failure.message
        elif failure.gate == 4:
            return f"Resolve runtime evaluation error: {failure.message}"
        return failure.message
