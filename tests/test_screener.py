"""Unit tests for the 2-Stage AST Screener (copilot_cli.screener)."""

from __future__ import annotations

import time
from pathlib import Path
import pytest

from copilot_cli.screener import (
    DEFAULT_IGNORED_DIRS,
    PATTERN_BASE_SETTINGS,
    PATTERN_CLASS_CONFIG,
    PATTERN_FIELD_REGEX,
    PATTERN_LEGACY_EXPORT,
    PATTERN_ON_EVENT,
    PATTERN_ROOT_VALIDATOR,
    PATTERN_VALIDATOR,
    FileScanResult,
    PatternMatch,
    scan_code,
    scan_directory,
    scan_file,
)


def test_detect_inner_class_config():
    """Detects inner 'class Config:' within a Pydantic model while ignoring top-level Config."""
    legacy_code = """
from pydantic import BaseModel

class UserModel(BaseModel):
    id: int
    name: str

    class Config:
        orm_mode = True
        allow_mutation = False
"""
    result = scan_code(legacy_code)
    assert result.has_legacy_patterns is True
    assert result.error is None
    assert len(result.matches) == 1
    match = result.matches[0]
    assert match.pattern_type == PATTERN_CLASS_CONFIG
    assert "class Config:" in match.snippet
    assert match.line == 8

    # Top-level class Config should NOT be flagged as legacy inner config
    top_level_config_code = """
class Config:
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
"""
    clean_result = scan_code(top_level_config_code)
    assert clean_result.has_legacy_patterns is False
    assert len(clean_result.matches) == 0


def test_detect_validator_decorator():
    """Detects legacy @validator decorators on model methods."""
    code = """
from pydantic import BaseModel, validator

class UserModel(BaseModel):
    username: str
    age: int

    @validator("username")
    def validate_username(cls, v):
        if len(v) < 3:
            raise ValueError("Too short")
        return v

    @validator("age", pre=True)
    def validate_age(cls, v):
        return int(v)
"""
    result = scan_code(code)
    assert result.has_legacy_patterns is True
    assert len(result.matches) == 2
    for m in result.matches:
        assert m.pattern_type == PATTERN_VALIDATOR
        assert "@validator" in m.snippet
    assert result.matches[0].line == 8
    assert result.matches[1].line == 14


def test_detect_root_validator_decorator():
    """Detects legacy @root_validator decorators."""
    code = """
from pydantic import BaseModel, root_validator

class Credentials(BaseModel):
    password: str
    password_confirm: str

    @root_validator
    def check_passwords(cls, values):
        if values.get("password") != values.get("password_confirm"):
            raise ValueError("Passwords do not match")
        return values

    @root_validator(pre=True)
    def clean_data(cls, values):
        return values
"""
    result = scan_code(code)
    assert result.has_legacy_patterns is True
    assert len(result.matches) == 2
    assert result.matches[0].pattern_type == PATTERN_ROOT_VALIDATOR
    assert result.matches[1].pattern_type == PATTERN_ROOT_VALIDATOR
    assert "@root_validator" in result.matches[0].snippet


def test_detect_legacy_export_dict_and_json():
    """Detects .dict() and .json() calls on objects/variables."""
    code = """
user = get_user()
user_data = user.dict()
user_clean = user.dict(exclude={"password"})
json_payload = user.json()
custom_json = user.json(indent=2)
"""
    result = scan_code(code)
    assert result.has_legacy_patterns is True
    assert len(result.matches) == 4
    for m in result.matches:
        assert m.pattern_type == PATTERN_LEGACY_EXPORT

    assert ".dict()" in result.matches[0].description
    assert ".dict()" in result.matches[1].description
    assert ".json()" in result.matches[2].description
    assert ".json()" in result.matches[3].description


def test_detect_base_settings_import():
    """Detects BaseSettings imported from pydantic."""
    code1 = """
from pydantic import BaseModel, BaseSettings, Field

class Settings(BaseSettings):
    api_key: str
"""
    res1 = scan_code(code1)
    assert res1.has_legacy_patterns is True
    assert len(res1.matches) == 1
    assert res1.matches[0].pattern_type == PATTERN_BASE_SETTINGS
    assert "BaseSettings" in res1.matches[0].snippet

    code2 = """
import pydantic

class AppSettings(pydantic.BaseSettings):
    debug: bool = False
"""
    res2 = scan_code(code2)
    assert res2.has_legacy_patterns is True
    assert any(m.pattern_type == PATTERN_BASE_SETTINGS for m in res2.matches)

    # Clean Pydantic v2 settings import should NOT be flagged
    modern_settings = """
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    port: int = 8000
"""
    res_modern = scan_code(modern_settings)
    assert res_modern.has_legacy_patterns is False
    assert len(res_modern.matches) == 0


def test_detect_app_on_event_decorator():
    """Detects legacy FastAPI @app.on_event('startup'/'shutdown') decorator calls."""
    code = """
from fastapi import FastAPI

app = FastAPI()

@app.on_event("startup")
async def startup_db_client():
    await db.connect()

@app.on_event("shutdown")
def shutdown_db_client():
    db.close()
"""
    result = scan_code(code)
    assert result.has_legacy_patterns is True
    assert len(result.matches) == 2
    assert result.matches[0].pattern_type == PATTERN_ON_EVENT
    assert "startup" in result.matches[0].description
    assert result.matches[1].pattern_type == PATTERN_ON_EVENT
    assert "shutdown" in result.matches[1].description


def test_detect_field_regex():
    """Detects Field(..., regex=...) keyword argument."""
    code = """
from pydantic import BaseModel, Field

class Account(BaseModel):
    username: str = Field(..., regex="^[a-zA-Z0-9_]{3,16}$")
    phone: str = Field(regex=r"^\\+?[0-9]{10,15}$")
    code: str = Field("default", description="code", regex="^[A-Z]{3}$")
"""
    result = scan_code(code)
    assert result.has_legacy_patterns is True
    assert len(result.matches) == 3
    for m in result.matches:
        assert m.pattern_type == PATTERN_FIELD_REGEX
        assert "Field(regex=...)" in m.description

    # Field with pattern=... in Pydantic v2 should NOT be flagged
    modern_field = """
from pydantic import BaseModel, Field

class ModernAccount(BaseModel):
    username: str = Field(..., pattern="^[a-zA-Z0-9_]{3,16}$")
"""
    clean_res = scan_code(modern_field)
    assert clean_res.has_legacy_patterns is False
    assert len(clean_res.matches) == 0


def test_false_positive_resistance():
    """Ensures benign patterns with similar names are not falsely flagged."""
    clean_tricky_code = """
import json
import re

# 1. Dictionary containing "dict" as a key
d = {"dict": 1, "json": 2}

# 2. Attribute access ending in _dict or _json
my_var = object()
name = my_var.dict_name
fmt = my_var.json_format

# 3. Clean function names
def json_loader(path: str) -> dict:
    with open(path) as f:
        return json.load(f)

def dict_converter(items: list) -> dict:
    return dict(items)

# 4. Standard built-in dict() and json module calls
empty_map = dict()
dumped = json.dumps({"status": "ok"})
loaded = json.loads('{"count": 10}')

# 5. Regex in standard library re.compile
pattern = re.compile(r"\\d+")

# 6. Event handlers that are regular methods, not decorators
class EventEmitter:
    def on_event(self, event_name: str):
        pass

emitter = EventEmitter()
emitter.on_event("click")
"""
    result = scan_code(clean_tricky_code)
    assert result.has_legacy_patterns is False
    assert len(result.matches) == 0
    assert result.error is None


def test_clean_file_detection():
    """Confirms that standard utility functions are marked as clean with zero overhead."""
    math_utils = """
import math
from typing import List

def euclidean_distance(p1: tuple[float, float], p2: tuple[float, float]) -> float:
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def fibonacci(n: int) -> List[int]:
    if n <= 0:
        return []
    fib = [0, 1]
    while len(fib) < n:
        fib.append(fib[-1] + fib[-2])
    return fib[:n]
"""
    result = scan_code(math_utils)
    assert result.has_legacy_patterns is False
    assert len(result.matches) == 0
    assert result.error is None


def test_stage_1_fast_rejection_timing():
    """Validates that Stage 1 rejects clean code in <0.01ms (<10 microseconds)."""
    clean_sample = """
def calculate_tax(subtotal: float, rate: float = 0.08) -> float:
    \"\"\"Compute total tax for an invoice line item.\"\"\"
    return round(subtotal * rate, 2)
"""
    # Warmup
    scan_code(clean_sample)

    start = time.perf_counter()
    iterations = 500
    for _ in range(iterations):
        res = scan_code(clean_sample)
        assert res.has_legacy_patterns is False
    elapsed = time.perf_counter() - start

    avg_time_ms = (elapsed / iterations) * 1000
    # Requirement: returns clean in <0.01ms
    assert avg_time_ms < 0.01, f"Expected <0.01ms, got {avg_time_ms:.5f}ms"


def test_syntax_error_handled_gracefully():
    """Handles syntax errors gracefully in code containing legacy tokens."""
    malformed_code = """
class UserModel:
    class Config:
        orm_mode =
"""
    result = scan_code(malformed_code)
    assert result.has_legacy_patterns is False
    assert result.error is not None
    assert "SyntaxError" in result.error


def test_scan_file(tmp_path: Path):
    """Verifies file scanning for legacy and clean files on disk."""
    legacy_file = tmp_path / "legacy.py"
    legacy_file.write_text(
        "from pydantic import BaseModel\n\nclass M(BaseModel):\n    class Config:\n        pass\n",
        encoding="utf-8",
    )

    clean_file = tmp_path / "clean.py"
    clean_file.write_text("x = 42\n", encoding="utf-8")

    nonexistent_file = tmp_path / "does_not_exist.py"

    res_legacy = scan_file(legacy_file)
    assert res_legacy.has_legacy_patterns is True
    assert len(res_legacy.matches) == 1

    res_clean = scan_file(clean_file)
    assert res_clean.has_legacy_patterns is False
    assert len(res_clean.matches) == 0

    res_missing = scan_file(nonexistent_file)
    assert res_missing.has_legacy_patterns is False
    assert res_missing.error is not None
    assert "File not found" in res_missing.error


def test_scan_directory_with_clean_and_legacy_and_ignored(tmp_path: Path):
    """Scans directories and verifies exclusion of .venv, __pycache__, .git, etc."""
    # 1. Clean root file
    (tmp_path / "helpers.py").write_text("def ping(): return 'pong'\n", encoding="utf-8")

    # 2. Legacy root file
    (tmp_path / "user.py").write_text(
        "from pydantic import BaseModel, validator\n\nclass U(BaseModel):\n    @validator('x')\n    def v(cls, val): return val\n",
        encoding="utf-8",
    )

    # 3. Subfolder with legacy file
    sub_dir = tmp_path / "api"
    sub_dir.mkdir()
    (sub_dir / "routes.py").write_text(
        "from fastapi import FastAPI\napp = FastAPI()\n@app.on_event('startup')\ndef start(): pass\n",
        encoding="utf-8",
    )

    # 4. Ignored directories containing legacy patterns that MUST NOT be scanned
    for ignored_name in [".venv", "__pycache__", ".git", ".pytest_cache", ".agents", "dist", "build"]:
        ignored_folder = tmp_path / ignored_name
        ignored_folder.mkdir()
        (ignored_folder / "legacy_in_ignored.py").write_text(
            "class Config:\n    pass\nuser.dict()\n",
            encoding="utf-8",
        )

    # Scan recursively
    results = scan_directory(tmp_path, recursive=True)
    scanned_filenames = [r.file_path.name for r in results]

    # Should only scan helpers.py, user.py, and routes.py
    assert sorted(scanned_filenames) == ["helpers.py", "routes.py", "user.py"]
    assert "legacy_in_ignored.py" not in scanned_filenames

    # Check findings
    findings = {r.file_path.name: r.has_legacy_patterns for r in results}
    assert findings["helpers.py"] is False
    assert findings["user.py"] is True
    assert findings["routes.py"] is True

    # Scan non-recursively
    non_rec_results = scan_directory(tmp_path, recursive=False)
    non_rec_filenames = [r.file_path.name for r in non_rec_results]
    assert sorted(non_rec_filenames) == ["helpers.py", "user.py"]
